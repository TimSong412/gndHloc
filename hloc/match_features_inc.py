import argparse
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
from queue import Queue
from threading import Thread
from functools import partial
from tqdm import tqdm
import h5py
import torch
import time
import threading

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 10,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    },
    'adalam': {
        'output': 'matches-adalam',
        'model': {
            'name': 'adalam'
        },
    }
}


class WorkQueue():
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,))
            for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k+'0'] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        with h5py.File(self.feature_path_r, 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k+'1'] = torch.from_numpy(v.__array__()).float()
            data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)


def main(conf: Dict,
         pairs: Path,
         features: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         overwrite: bool = False) -> Path:

    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features+'.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     overwrite: bool = False) -> Path:
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True)
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=.1)):
        data = {k: v if k.startswith('image')
                else v.to(device, non_blocking=True) for k, v in data.items()}
        pred = model(data)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info('Finished exporting matches.')


class FeaturePairsIncDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_q, feature_r, device):
        self.pairs = pairs
        self.feature_q = {}
        self.feature_r= feature_r
        for k, v in feature_q.items():
            tempdict = {}
            for tk, tv in v.items():
                tempdict[tk] = torch.from_numpy(tv).float().to(device=device)
                if tk == "image_size":
                    tempdict["image_size_cpu"] = tv
            self.feature_q[k] = tempdict

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        grp_q = self.feature_q[name0]
        for k, v in grp_q.items():
            # data[k+'0'] = torch.from_numpy(v).float()
            data[k+'0'] = v
            if k == 'image_size_cpu':
                continue
        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1,)+tuple(grp_q['image_size_cpu'])[::-1])

        grp_r = self.feature_r[name1]
        for k, v in grp_r.items():
            # data[k+'1'] = torch.from_numpy(v).float()
            if k == 'image_size_cpu':
                continue
            data[k+'1'] = v
        data['image1'] = torch.empty((1,)+tuple(grp_r['image_size_cpu'])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


class FeatureMatcher():
    def __init__(self, 
                 conf: Dict,
                 num_globalmatch,
                 features_ref: Optional[Path] = None) -> None:

        logger.info('Matching local features with configuration:'
                    f'\n{pprint.pformat(conf)}')

        if not features_ref.exists():
            raise FileNotFoundError(f'Reference feature file {features_ref}.')
        # match_path.parent.mkdir(exist_ok=True, parents=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_ref = self.readfeature(features_ref)
        
        # Model = dynamic_load(matchers, conf['model']['name'])
        self.models = []
        self.num_models = num_globalmatch
        for i in range(num_globalmatch):
            Model = dynamic_load(matchers, conf['model']['name'])
            self.models.append(Model(conf['model']).eval().to(self.device))
        # self.model = Model(conf['model']).eval().to(self.device)
        self.preds= [None for i in range(num_globalmatch)]
        self.result = {}
        self.resultlock = threading.Lock()
        self.datalock = threading.Lock()
        self.worknumber = 2
        self.modellocks = [threading.Lock() for i in range(self.worknumber)]        

    def readfeature(self, featurepth):
        feats = {}
        with h5py.File(featurepth, 'r') as fd:
            for name, data in fd.items():
                datadict = {}
                for key, value in data.items():
                    datadict[key] = torch.from_numpy(value.__array__()).float().to(self.device, non_blocking=True)
                    if key == "image_size":
                        datadict["image_size_cpu"] = value.__array__()
                feats[name] = datadict
        return feats

    @torch.no_grad()
    def match(self, pairs, feature_q):
        # pairs = parse_retrieval(pairs_path)
        # pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        # pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
        if len(pairs) == 0:
            logger.info('Skipping the matching.')
            return
        
        dataset = FeaturePairsIncDataset(pairs, feature_q, self.feature_ref, self.device)
        loader = torch.utils.data.DataLoader(
            dataset, num_workers=0, batch_size=1, shuffle=False)
        # writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)
        st = time.time()
        inf_threads = []
        t0 = time.time()
        for idx, data in enumerate(loader):#, smoothing=.1)):  
            self.datalock.acquire()          
            data = {k: v if k.startswith('image')
                    else v.to(self.device, non_blocking=True) for k, v in data.items()}   
            # print("datatime= ", time.time()-t0, " tid= ", idx)         
            pair = names_to_pair(*pairs[idx])
            inft = Thread(target=self.inference, args=[data, pair, idx])
            inf_threads.append(inft)
                     
            inft.start()
            print("launchtime= ", time.time()-t0, " tid= ", idx) 
            # t0 = time.time()
            # pred = self.model(data)                                
            # self.add_pair(pair, pred)
        for inft in inf_threads:
            inft.join()
        print("all_match_time = ", time.time()-st)
        # writer_queue.join()
        logger.info('Finished exporting matches.')
        return self.result

    def add_pair(self, pair, pred):
        # grp = fd.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp = {'matches0': matches}
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp['matching_scores0'] = scores
        self.result[pair] = grp
        self.resultlock.release()
        
    @torch.no_grad()
    def inference(self, data, pair, tid):        
        
        self.datalock.release()   
        self.modellocks[tid%self.worknumber].acquire()
        t0 = time.time()
        self.preds[tid] = self.models[tid%self.num_models](data)        
        self.modellocks[tid%self.worknumber].release()
        # self.modellock.release()
        # return
        self.resultlock.acquire()
        self.add_pair(pair, self.preds[tid])
        # print("inference&add_time= ", time.time()-t0, "model id= ", tid)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
