/* License: MIT */

#include <cstdlib>
#include <cstring>
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


uint32_t predict_probs(fasttext::FastText* fasttext, int32_t k, fasttext::real threshold, prediction* predictions, char* input, uint32_t len) {
  membuf sbuf(input, input + len);
  std::istream in(&sbuf);
  std::vector<std::pair<fasttext::real, std::string>> ftPredictions;
  fasttext->predictLine(in, ftPredictions, k, threshold);
  for(size_t i = 0; i < ftPredictions.size(); i++) {
      predictions[i].score = ftPredictions[i].first;
      predictions[i].label_size = (uint32_t)ftPredictions[i].second.size();
      strcpy(predictions[i].label, ftPredictions[i].second.c_str());
  }
  return ftPredictions.size();
}

fasttext::real predict_best(fasttext::FastText* fasttext, fasttext::real threshold, char* label, char* input, uint32_t len) {
  membuf sbuf(input, input + len);
  std::istream in(&sbuf);
  std::vector<std::pair<fasttext::real, std::string>> ftPredictions;
  fasttext->predictLine(in, ftPredictions, 1, threshold);
  for(size_t i = 0; i < ftPredictions.size(); i++) {
      memcpy(label, ftPredictions[i].second.c_str(), ftPredictions[i].second.size() + 1);
      return ftPredictions[i].first;
  }
  return 0;
}

int get_dimension(fasttext::FastText* fasttext) {
    return fasttext->getDimension();
}

/* TODO: vector constructor + printer, so we can fasttext.getWordVector */

}
