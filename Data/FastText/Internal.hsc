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
import qualified Data.ByteString as S
import Data.ByteString.Internal (memcpy, ByteString(..))
import Control.Exception(mask_)

-- | A single predicted label with probability.
data Prediction = Prediction { pScore:: Float, pLabel :: ByteString }
  deriving (Show, Eq)
instance Storable Prediction where
  alignment _ = #alignment struct prediction
  sizeOf _    = #size struct prediction
  peek ptr    = do
      sc <- (#peek struct prediction, score) ptr
      len <- (#peek struct prediction, label_size) ptr :: IO Word32
      str <- (#peek struct prediction, label) ptr
      lb <- S.packCStringLen (str, fromIntegral len)
      pure (Prediction sc lb)
  poke ptr (Prediction sc lb@(PS fp 0 len)) = do
    buf <- mallocBytes (len+1)   -- based off useAsCStringLen
    withForeignPtr fp $ \p -> do -- but malloc instead of alloca
      memcpy buf p len           -- so we don't free after use
      pokeByteOff buf len (0::Word8)
      let str = castPtr buf
      (#poke struct prediction, score     ) ptr sc
      (#poke struct prediction, label_size) ptr len
      (#poke struct prediction, label     ) ptr str

-- | A fastText model loaded into memory (see 'Data.FastText.loadModel').
data Model

-- | A fastText probability score.
type FTReal = CFloat

-- | Load a fasttext model from file.
-- Internal use only.
foreign import ccall unsafe "load_model" loadModel :: CString -> IO (Ptr Model)
-- | Use a fasttext model to predict (top @k@) labels of @input@.
-- Internal use only.
foreign import ccall unsafe "predict_probs" predictProbs :: Ptr Model -> Int -> Float -> Ptr Prediction -> CString -> Word32 -> IO Int
foreign import ccall unsafe "predict_best" predictBest :: Ptr Model -> Float -> CString -> CString -> Word32 -> IO FTReal

-- | Get the dimensions of a loaded fastText model.
foreign import ccall unsafe "get_dimension" getDimension :: Ptr Model -> Int
