{-# LANGUAGE ForeignFunctionInterface #-}

#include "extern.h"
#include <stddef.h>

{-|
Module      : Data.FastText.Internal
Description : Raw/internal FFI bindings to the fastText library.
Copyright   : (c) Kevin Brubeck Unhammer, 2022
License     : BSD-2-Clause
Maintainer  : kevin@trigram.no
Stability   : experimental
Portability : POSIX

Internal FFI bindings, API subject to change.
-}

module Data.FastText.Internal where

import System.IO.Unsafe
import Foreign
import Foreign.C
import Foreign.Marshal.Array
import Data.ByteString (ByteString)
import qualified Data.ByteString as S
import Control.Exception(mask_)

-- | A single predicted label with probability.
data Prediction = Prediction { pScore:: Float, pLabel :: ByteString }
  deriving (Show, Eq)
instance Storable Prediction where
  alignment _ = #alignment struct prediction
  sizeOf _    = #size struct prediction
  peek ptr    = do
      sc <- (#peek struct prediction, score) ptr
      len <- (#peek struct prediction, label_size) ptr
      str <- (#peek struct prediction, label) ptr
      lb <- S.packCStringLen (str, len)
      pure (Prediction sc lb)
  poke ptr (Prediction sc lb) = S.useAsCStringLen lb $ \(str, len) -> do
      (#poke struct prediction, score     ) ptr sc
      (#poke struct prediction, label_size) ptr str
      (#poke struct prediction, label     ) ptr len

-- | A fastText model loaded into memory (see 'Data.FastText.loadModel').
data Model

-- | A fastText probability score.
type FTReal = CFloat

-- | Load a fasttext model from file.
-- Internal use only.
foreign import ccall unsafe "load_model" loadModel :: CString -> IO (Ptr Model)
-- | Use a fasttext model to predict (top @k@) labels of @input@.
-- Internal use only.
foreign import ccall unsafe "predict_probs" predictProbs :: Ptr Model -> Int -> Float -> Ptr Prediction -> CString -> Int -> IO Int

-- | Get the dimensions of a loaded fastText model.
foreign import ccall unsafe "get_dimension" getDimension :: Ptr Model -> Int


-- vector.h
data Vector
data Matrix
foreign import ccall unsafe "vector_data" vector_data :: Ptr Vector -> Ptr FTReal
foreign import ccall unsafe "vector_at" vector_at :: Ptr Vector -> Int64 -> FTReal
foreign import ccall unsafe "vector_size" vector_size :: Ptr Vector -> Int64
foreign import ccall unsafe "vector_zero" vector_zero :: Ptr Vector -> IO ()
foreign import ccall unsafe "vector_scale" vector_scale :: Ptr Vector -> FTReal -> IO ()
foreign import ccall unsafe "vector_norm" vector_norm :: Ptr Vector -> FTReal
foreign import ccall unsafe "vector_addVector" vector_addVector :: Ptr Vector -> Ptr Vector -> IO ()
foreign import ccall unsafe "vector_addVectorScaled" vector_addVectorScaled :: Ptr Vector -> Ptr Vector -> FTReal -> IO ()
foreign import ccall unsafe "vector_addRow" vector_addRow :: Ptr Vector -> Ptr Matrix -> Int64 -> IO ()
foreign import ccall unsafe "vector_addRowScaled" vector_addRowScaled :: Ptr Vector -> Ptr Matrix -> Int64 -> FTReal -> IO ()
foreign import ccall unsafe "vector_mul" vector_mul :: Ptr Vector -> Ptr Matrix -> Ptr Vector -> IO ()
foreign import ccall unsafe "vector_argmax" vector_argmax :: Ptr Vector -> IO Int64


-- | Get a word vector
foreign import ccall unsafe "get_word_vector" getWordVector :: Ptr Model -> CString -> IO (Ptr Vector)
