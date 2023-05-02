import numpy as np
from PIL import Image

from modules.utils.utils import overlap_ratio


class SampleGenerator():
    def __init__(self, type_, img_size, trans=1, scale=1, aspect=None, valid=False):
        self.type = type_
        self.img_size = np.array(img_size)  # The size of original image, (w, h)
        self.trans = trans
        self.scale = scale
        self.aspect = aspect
        self.valid = valid

    def _gen_samples(self, bbox, n):
        
        # bbox: target bbox (min_x,min_y,w,h)
        bbox = np.array(bbox, dtype='float32')

        # (min_x,min_y,w,h) --> (center_x, center_y, w, h)
        sample = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]], dtype='float32')
        # sample.shape=(4,) ->sample[None, :].shape=(1,4) -> samples.shape=(n,4)
        samples = np.tile(sample[None, :], (n, 1))  

        # various aspect ratio
        if self.aspect is not None:
            ratio = np.random.rand(n, 2) * 2 - 1
            samples[:, 2:] *= self.aspect ** ratio

        # sample generation
        if self.type == 'gaussian':
            samples[:, :2] += self.trans * np.mean(bbox[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

        elif self.type == 'uniform':
            # samples[:, :2] are (center_x, center_y), sample location random transform or move
            samples[:, :2] += self.trans * np.mean(bbox[2:]) * (np.random.rand(n, 2) * 2 - 1)
            # samples[:, 2:] are (w, h), sample size random scale
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == 'whole':
            m = int(2 * np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bbox[2:] / 2 + xy * (self.img_size - bbox[2:] / 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range, limiting the bbox size to between 10 and img_size-10
        samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        if self.valid:
            samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
        else:
            # limiting the bbox center position to between 0 and img_size
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        # (center_x, center_y, w, h) --> (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2

        return samples

    def __call__(self, bbox, n, overlap_range=None, scale_range=None):

        if overlap_range is None and scale_range is None:
            return self._gen_samples(bbox, n)

        else:
            samples = None
            remain = n  # ?
            factor = 2  # ?
            while remain > 0 and factor < 16:
                _samples = self._gen_samples(bbox, remain * factor)

                idx = np.ones(len(_samples), dtype=bool)  # generate full-one list
                if overlap_range is not None:
                    # Compute overlap ratio (IoU) between '_samples' and 'bbox'
                    iou = overlap_ratio(_samples, bbox)
                    # Filter 'index' of '_samples' whose IoU is between overlap thresholds
                    idx *= (iou >= overlap_range[0]) * (iou <= overlap_range[1])
                if scale_range is not None:
                    # s = the areas of '_samples' / the areas of 'bbox'
                    s = np.prod(_samples[:, 2:], axis=1) / np.prod(bbox[2:])
                    idx *= (s >= scale_range[0]) * (s <= scale_range[1])
                _samples = _samples[idx, :]  # Filter '_samples' by 'index
                # Save the first 'min(remain, len(_samples))' '_samples'
                _samples = _samples[:min(remain, len(_samples))] 
                if samples is None:
                    samples = _samples
                else:
                    samples = np.concatenate([samples, _samples])
                remain = n - len(samples)
                factor = factor * 2

            return samples

    def set_type(self, type_):
        self.type = type_

    def set_trans(self, trans):
        self.trans = trans

    def expand_trans(self, trans_limit):
        self.trans = min(self.trans * 1.1, trans_limit)
