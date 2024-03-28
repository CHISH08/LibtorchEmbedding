#include "embedding.hpp"
#include <unistd.h>
#include <codecvt>
#include <locale>

int main() {
    short num_cores = sysconf(_SC_NPROCESSORS_CONF);
    int vocab_size = 8;
    short embedding_dim = 20, window_size = 1, batch_size = 100, num_epochs = 100;
    double lr = 0.01;
    std::locale::global(std::locale(""));

    std::string device = "cpu";
    Embedding<CBOW> model(vocab_size, window_size, embedding_dim, lr, device);
    std::u32string text = U"Мама мыла раму, а папа ломал раму";
    torch::Tensor idxs = model.text_to_idx(text);
    model.fit(idxs, batch_size, num_epochs);
    std::u32string test_word = U"мама";
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    std::string test_word_utf8 = convert.to_bytes(test_word);
    std::cout << test_word_utf8 << ":\n";
    std::cout << model[test_word] << std::endl;
    return 0;
}
