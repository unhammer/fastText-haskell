/* License: MIT */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*
 fasttext returns C++ std::pair's, but we need plain C structs in order to use peekArray (Storable)

 Using std::pair might be possible, but costly according to #haskell:
 Unhammer | How would I create a Storable instance for a c++ std::pair<float,std::string> ? Is it as if it were a struct?
 #haskell | Unhammer: You cry
 #haskell | Unhammer: And take up drinking
 #haskell | I suspect you need to export "C" it as a struct
 #haskell | then you first have to decide how in the world to get something out of that std::string
 #haskell | Unhammer: I highly recommend Laphraoig 10
*/
struct prediction {
  float score;                  /* fasttext::real */
  uint32_t label_size;
  char *label;
};


#ifdef __cplusplus
}
#endif

