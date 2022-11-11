# fasttext bindings for Haskell

This library lets you `loadModel` and `predictProbs` from Haskell.

Example usage:

```haskell
{-# LANGUAGE OverloadedStrings            #-}

module Main where

import Data.FastText

main :: IO ()
main = do
  model <- loadModel "data/lid.176.ftz"
  let input = "Une liste d'infinitifs prolonge l'accident."
  res <- predictProbs model 2 0 input
  print res
```

Output:

    [Prediction {pScore = 0.9801573, pLabel = "__label__fr"},Prediction {pScore = 1.0023363e-2, pLabel = "__label__zh"}]

# Related libraries

- [python bindings to fastText](https://fasttext.cc/docs/en/python-module.html)
- [hastext](https://github.com/nnwww/hastext) – pure Haskell implementation of fastText
- [word2vec-model](https://hackage.haskell.org/package/word2vec-model) – Haskell bindings to word2vec
- [hs-word2vec](https://github.com/abailly/hs-word2vec) – pure Haskell implementation of word2vec

# Vendored stuff

The files in cbits except `extern.{cc,h}` are from
https://github.com/facebookresearch/fastText/commit/3697152e0fd772d9185697fdbd4a1d340ca5571d

Test model in `data/lid.176.ftz` is CC-SA 3.0 from
https://fasttext.cc/docs/en/language-identification.html
