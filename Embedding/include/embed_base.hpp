#pragma once
#include <torch/torch.h>

class Model : public torch::nn::Module {
public:
    virtual ~Model() = default;
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual torch::Tensor operator()(torch::Tensor input) = 0;
    virtual std::pair<std::vector<double>, std::vector<double>> fit(torch::Tensor &data, short batch_size, int64_t num_epochs, size_t num_workers) = 0;
    std::string device;
    short window_size;
};
