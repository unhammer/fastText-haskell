{-# LANGUAGE OverloadedStrings            #-}
{-# LANGUAGE ForeignFunctionInterface #-}

{-|
Module      : Data.FastText
Description : Public bindings to the fastText library.
Copyright   : (c) Kevin Brubeck Unhammer, 2022
License     : BSD-2-Clause
Maintainer  : kevin@trigram.no
Stability   : experimental
Portability : POSIX

This module lets you load a fastText model from a file, and use it to
get label predictions.

If you did preprocessing on text when training the model, remember to
preprocess your input in the same way.
-}

module Data.FastText (Prediction(..), getDimension, Model, loadModel, predictProbs, predictBest) where

import qualified Data.ByteString as S
import System.IO.Unsafe
import Foreign
import Foreign.C
import Foreign.Marshal.Array
import Control.Exception(mask_)
-- import Foreign.Utilities

import qualified Data.FastText.Internal as FFI
import Data.FastText.Internal (Model, getDimension, Prediction(..))

-- | Load a fasttext model from file.
loadModel :: FilePath -> IO (Ptr Model)
loadModel path = withCString path $ \cpath -> FFI.loadModel cpath

-- | Use a fasttext model to predict (top @k@) labels of @input@.
-- Best predictions first. Only includes those where score is over @threshold@.
--
-- Use 'loadModel' to get a @model@. To get just the top prediction, use @k@ = 1.
-- If @threshold@ is 0.0, all @k@ predictions are included, otherwise you may get less than @k@.
-- You may also see less than @k@ predictions if @k@ > number of labels in model.
predictProbs :: Ptr Model -> Int -> Float -> S.ByteString -> IO [Prediction]
predictProbs model k threshold input = S.useAsCStringLen input $ \(cinput, clen) -> do
    let emptyPrediction = Prediction 0 ""
    withArray (take k $ repeat emptyPrediction) $ \preds -> do
      actuallyPredicted <- FFI.predictProbs model k threshold preds cinput (fromIntegral clen)
      res <- peekArray k preds
      pure $ res

maxLabelLength = 256       -- 256b should be enough for everybody

predictBest :: Ptr Model -> Float -> S.ByteString -> IO Prediction
predictBest model threshold input = S.useAsCStringLen input $ \(cinput, clen) ->
    allocaBytes maxLabelLength $ \clabel -> do
      cscore <- FFI.predictBest model threshold clabel cinput (fromIntegral clen)
      label <- S.packCString clabel
      let score = realToFrac cscore
      pure $ Prediction score label
