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
      p.label_size = ftPredictions[i].second.size();
      p.label = ftPredictions[i].second.c_str();
      predictions[i] = p;
  }
  return ftPredictions.size();
}

int get_dimension(fasttext::FastText* fasttext) {
    return fasttext->getDimension();
}

// vector.h
const fasttext::real* vector_data(const fasttext::Vector* self) {
    return self->data();
}
fasttext::real vector_at(const fasttext::Vector* self, int64_t i) {
    return (*self)[i];
}
int64_t vector_size(fasttext::Vector* self) {
    return self->size();
}
void vector_zero(fasttext::Vector* self) {
    self->zero();
}
void vector_scale(fasttext::Vector* self, fasttext::real a) {
    self->mul(a);
}
fasttext::real vector_norm(const fasttext::Vector* self) {
    return self->norm();
}
void vector_addVector(fasttext::Vector* self, const fasttext::Vector* source) {
    self->addVector(*source);
}
void vector_addVectorScaled(fasttext::Vector* self, const fasttext::Vector* source, fasttext::real s) {
    self->addVector(*source, s);
}
void vector_addRow(fasttext::Vector* self, const fasttext::Matrix* A, int64_t i) {
    self->addRow(*A, i);
}
void vector_addRowScaled(fasttext::Vector* self, const fasttext::Matrix* A, int64_t i, fasttext::real a) {
    self->addRow(*A, i, a);
}
void vector_mul(fasttext::Vector* self, const fasttext::Matrix* A, const fasttext::Vector* vec) {
    self->mul(*A, *vec);
}
int64_t vector_argmax(fasttext::Vector* self) {
    return self->argmax();
}

/* TODO: vector constructor + printer, so we can fasttext.getWordVector */

}
