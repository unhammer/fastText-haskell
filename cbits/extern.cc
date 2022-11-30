/* License: MIT */

#include <cstdlib>
#include "extern.h"
#include "fasttext.h"

extern "C" {

fasttext::FastText *load_model(char *path) {
  fasttext::FastText *fasttext = new fasttext::FastText();
  fasttext->loadModel(std::string(path));
  return fasttext;
}

/* https://stackoverflow.com/a/7782037/69663 */
struct membuf : std::streambuf
{
    membuf(char* begin, char* end) {
        this->setg(begin, begin, end);
    }
};


size_t predict_probs(fasttext::FastText* fasttext, int32_t k, fasttext::real threshold, prediction* predictions, char* input, size_t len) {
  bool printProb = true;
  membuf sbuf(input, input + len);
  std::istream in(&sbuf);
  std::vector<std::pair<fasttext::real, std::string>> ftPredictions;
  fasttext->predictLine(in, ftPredictions, k, threshold);
  for(size_t i = 0; i < ftPredictions.size(); i++) {
      prediction p;
      p.score = ftPredictions[i].first;
      p.label_size = (uint32_t)ftPredictions[i].second.size();
      p.label = ftPredictions[i].second.c_str();
      predictions[i] = p;
  }
  return ftPredictions.size();
}

int get_dimension(fasttext::FastText* fasttext) {
    return fasttext->getDimension();
}

/* TODO: vector constructor + printer, so we can fasttext.getWordVector */

}
