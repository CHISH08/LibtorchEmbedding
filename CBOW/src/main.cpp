// main.cpp
#include "lib.hpp"
#include <unistd.h>

int main() {
    short num_cores = sysconf(_SC_NPROCESSORS_CONF);
    int vocab_size = 8;
    short embedding_dim = 20, window_size = 2, batch_size = 100, num_epochs = 10000;
    double lr = 0.01;

    std::string device = "cuda";
    CBOW cbow(vocab_size, window_size, embedding_dim, lr, device);
    std::string text = "Мама мыла раму, а папа мыл посуду!";
    torch::Tensor idx = torch::tensor({0, 1, 2, 3, 4, 5, 1, 6, 7});
    cbow.fit(idx, batch_size, num_epochs, num_cores);
}
