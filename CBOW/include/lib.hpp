#pragma once
#include <torch/torch.h>
#include <iostream>

class CBOW: public torch::nn::Module {
public:
    CBOW(int64_t vocab_size, short window_size, int32_t embedding_dim, float lr, std::string device);
    torch::Tensor operator()(torch::Tensor input);
    ~CBOW() = default;

    torch::Tensor forward(torch::Tensor x);
    void fit(torch::Tensor data, short batch_size, int64_t num_epochs, size_t num_workers);
private:
    torch::nn::Embedding embeddings{nullptr};
    torch::nn::Linear linear{nullptr};
    std::unique_ptr<torch::optim::Optimizer> optim;
    short window_size;
    int64_t vocab_size;
    double lr;
    std::string device;
};
