import torch
import numpy
import time
from copy import deepcopy

__all__ = ["Prune"]


class Prune:
    def __init__(
        self,
        model,
        prune_type: int = 0,
        pretrain_step: int = 0,
        sparse_step: int = 0,
        current_step: int = 0,
        frequency: int = 100,
        prune_dict: dict = {},
        restore_sparsity: bool = False,
        fix_sparsity: bool = False,
        prune_device: str = "default",
        mask=None,
        sample_count: int = 400, # deprecated
        sample_ratio: float = 0.005,
        sample_layer: str = "attention.output.dense.weight", # deprecated
        logger=None
    ):
        self._model = model
        self._sample_count = sample_count 
        self._sample_ratio = sample_ratio
        self._sample_layer = sample_layer 
        self._prune_type = prune_type
        self._t = current_step 
        self._pretrain_step = pretrain_step
        self._sparse_step = sparse_step
        self._frequency = frequency
        self._prune_dict = prune_dict
        self._restore_sparsity = restore_sparsity
        self._fix_sparsity = fix_sparsity
        self._prune_device = prune_device
        self._variance = 0
        self._check_parameter()
        if mask:
            self._mask = mask
            self._fix_sparsity = True
            self.prune()
        else:
            self._mask = {}
            self._prepare()

        self._logger = logger

    def _check_parameter(self):
        assert isinstance(self._pretrain_step, int)
        assert isinstance(self._sparse_step, int)
        assert isinstance(self._frequency, int)
        assert isinstance(self._prune_dict, dict)
        assert isinstance(self._restore_sparsity, bool)
        assert isinstance(self._fix_sparsity, bool)
        assert self._prune_device in ["default", "cpu"]

    def _prepare(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if self._restore_sparsity == True:
                        mask = torch.where(
                            weight == 0,
                            torch.zeros_like(weight),
                            torch.ones_like(weight),
                        )
                        self._mask[name] = mask
                    else:
                        self._mask[name] = torch.ones_like(weight)

    def _update_mask(self, name, weight, keep_k, current_sparsity=-1):
        if keep_k >= 1:
            if self._prune_type == 0:
                reshape_weight = weight.reshape(-1)
                index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
                mask = numpy.zeros(reshape_weight.shape)
                mask[index] = 1
                mask = mask.reshape(weight.shape)
                mask = torch.as_tensor(mask, dtype=weight.dtype, device=weight.device)
                self._mask[name][:] = mask
            elif self._prune_type == 1: 
                # based on # of saved weights * sample_ratio 
                reshape_weight = weight.cpu().reshape(-1)
                index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
                mask_determinstic = numpy.zeros(reshape_weight.shape)
                mask_determinstic[index] = 1
                
                reshape_weight = reshape_weight ** 5
                weight_probality = abs(reshape_weight) / sum(abs(reshape_weight))

                masks = torch.zeros_like(reshape_weight)  
                sample_times = max(int(keep_k * self._sample_ratio), 1)
                for i in range(sample_times):
                    sampling_mask = weight_probality.multinomial(num_samples=keep_k, replacement=False)
                    masks[sampling_mask] += 1
                index = torch.topk(masks, keep_k)[1]
                masks[:] = 0
                masks[index] = 1
                self.calculate_variance(mask_determinstic, masks)

                mask = masks.reshape(weight.shape).to(weight.device)
                self._mask[name][:] = mask        
            elif self._prune_type == 2: 
                # based on # of removed weights] * sample_ratio
                reshape_weight = weight.cpu().reshape(-1)
                index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
                mask_determinstic = numpy.zeros(reshape_weight.shape)
                mask_determinstic[index] = 1

                reshape_weight = reshape_weight ** 5 
                weight_probality = abs(reshape_weight) / sum(abs(reshape_weight))

                masks = torch.zeros_like(reshape_weight)  
                sample_times = max(int((len(reshape_weight) - keep_k) * self._sample_ratio), 1)
                for i in range(sample_times):
                    sampling_mask = weight_probality.multinomial(num_samples=keep_k, replacement=False)
                    masks[sampling_mask] += 1
                index = torch.topk(masks, keep_k)[1]
                masks[:] = 0
                masks[index] = 1 
                self.calculate_variance(mask_determinstic, masks)

                mask = masks.reshape(weight.shape).to(weight.device)
                self._mask[name][:] = mask
            elif self._prune_type == 3: 
                # random
                reshape_weight = weight.cpu().reshape(-1)
                index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
                mask_determinstic = numpy.zeros(reshape_weight.shape)
                mask_determinstic[index] = 1

                mask = torch.zeros_like(reshape_weight)
                index = torch.randperm(len(mask))[:keep_k]
                mask[:] = 0
                mask[index] = 1 
                self.calculate_variance(mask_determinstic, mask)

                mask = mask.reshape(weight.shape).to(weight.device)
                self._mask[name][:] = mask
            elif self._prune_type == 4: 
                reshape_weight = weight.cpu().reshape(-1)
                index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
                mask_determinstic = numpy.zeros(reshape_weight.shape)
                mask_determinstic[index] = 1

                weight_probality = abs(reshape_weight) / sum(abs(reshape_weight))
                masks = torch.zeros_like(reshape_weight)  

                if weight.shape == (768, 768): 
                    self._sample_count = 27
                elif weight.shape == (3072, 768) or weight.shape == (768, 3072):
                    self._sample_count = 110 

                for i in range(self._sample_count):
                    sampling_mask = weight_probality.multinomial(num_samples=keep_k, replacement=False)
                    masks[sampling_mask] += 1
                index = torch.topk(masks, keep_k)[1]
                masks[:] = 0
                masks[index] = 1 
                self.calculate_variance(mask_determinstic, masks)

                mask = masks.reshape(weight.shape).to(weight.device)
                self._mask[name][:] = mask  
            elif self._prune_type == 5: 
                # a variant
                reshape_weight = weight.reshape(-1)
                index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
                mask_determinstic = numpy.zeros(reshape_weight.shape)
                mask_determinstic[index] = 1

                top_kth_value = reshape_weight.abs()[index[-1]]
                up_bound_value = top_kth_value * 3/2 

                temp_b = torch.ones_like(reshape_weight) 
                temp_b[reshape_weight.abs() < top_kth_value] = 0
                temp_b[reshape_weight.abs() > up_bound_value] = 0
                left_num_candidate = int(temp_b.sum().item())

                new_top_k = keep_k + left_num_candidate 
                num_candidate = left_num_candidate * 2

                new_index = torch.topk(reshape_weight.abs(), new_top_k)[1].cpu().numpy().tolist()                
                new_top_k_weight = torch.topk(reshape_weight.abs(), new_top_k)[0].cpu().numpy().tolist()
                candidate_weights = torch.as_tensor(new_top_k_weight[-num_candidate:], dtype=torch.float32) ** 5 

                weight_probality = candidate_weights.abs() / candidate_weights.abs().sum()
                masks = torch.zeros_like(candidate_weights)  
                sample_times = max(int(left_num_candidate * self._sample_ratio), 1)
                for i in range(sample_times):
                    sampling_mask = weight_probality.multinomial(num_samples=left_num_candidate, replacement=False)
                    masks[sampling_mask] += 1
                index = torch.topk(masks, left_num_candidate)[1]

                random_position = torch.as_tensor(new_index[-num_candidate:], dtype=torch.int32)[index].numpy().tolist()
                new_index = new_index[:-num_candidate] + random_position 

                mask = numpy.zeros(reshape_weight.shape)
                mask[new_index] = 1
                self.calculate_variance(mask_determinstic, mask)

                mask = mask.reshape(weight.shape)
                mask = torch.as_tensor(mask, dtype=weight.dtype, device=weight.device)
                self._mask[name][:] = mask
                
            elif self._prune_type == 6: 
                #  a variant
                reshape_weight = weight.cpu().reshape(-1)
                deterministic_index = torch.topk(reshape_weight.abs(), keep_k - 3000)[1].cpu().numpy().tolist()

                reshape_weight = reshape_weight ** 30 
                weight_probality = abs(reshape_weight) / sum(abs(reshape_weight))

                masks = torch.zeros_like(reshape_weight)  
                sample_times = max(int((len(reshape_weight) - keep_k) * self._sample_ratio), 1)
                for i in range(sample_times):
                    sampling_mask = weight_probality.multinomial(num_samples=keep_k, replacement=False)
                    masks[sampling_mask] += 1
                    masks[deterministic_index] += 1
                index = torch.topk(masks, keep_k)[1]
                masks[:] = 0
                masks[index] = 1 

                mask = masks.reshape(weight.shape).to(weight.device)
                self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def calculate_variance(self, mask1, mask2):
        if not isinstance(mask1, torch.Tensor):
            mask1 = torch.from_numpy(mask1)
        else:
            mask1 = mask1.cpu()
        
        if not isinstance(mask2, torch.Tensor):
            mask2 = torch.from_numpy(mask2)
        else:
            mask2 = mask2.cpu()

        same = (mask1 == mask2).int().sum()
        same_one_count = ((mask1 + mask2) == 2).int().sum()
        same_zero_count = ((mask1 + mask2) == 0).int().sum()
        not_same_count = mask1.numpy().size - same
        variance = not_same_count.item() / len(mask1)
        
        print(same, same_one_count, same_zero_count, not_same_count, variance, "\n")
        self._variance += variance 

    def _get_weight(self, parameter):
        if self._prune_device == "default":
            weight = parameter.data
        elif self._prune_device == "cpu":
            weight = parameter.data.to(device=torch.device("cpu"))
        return weight

    def prune(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if not self._fix_sparsity:
                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        keep_k = int(
                            weight.cpu().numpy().size * (1.0 - target_sparsity)
                        )
                        
                        start_time = time.time()
                        self._update_mask(name, weight, keep_k)
                        self._logger.info(f"target_sparsity: {target_sparsity}, name: {name}, shape: {weight.shape}, keep_k: {keep_k}, duration: {time.time() - start_time}")
                    parameter.mul_(self._mask[name]) 
        self._fix_sparsity = True

    def get_layer_sparsity(self):
        for name, parameter in self._model.named_parameters():
            if any(name == one for one in self._prune_dict):
                temp = parameter.data.cpu().numpy()
                curr_sparsity = 1 - numpy.flatnonzero(temp).size / temp.size
                return curr_sparsity 

    def prune_sparsity(self):
        total_param = 0
        total_nonezero = 0
        layer_sparse_rate = {}

        for name, parameter in self._model.named_parameters():
            if any(name == one for one in self._prune_dict):
                temp = parameter.data.cpu().numpy()
                total_param = total_param + temp.size
                total_nonezero = total_nonezero + numpy.flatnonzero(temp).size
                layer_sparse_rate[name] = 1 - numpy.flatnonzero(temp).size / temp.size
        total_sparse_rate = 1 - total_nonezero / total_param
        return layer_sparse_rate, total_sparse_rate