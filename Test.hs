{-# LANGUAGE OverloadedStrings            #-}

module Main where

import Data.FastText
import qualified Data.FastText.Internal as FFI
import System.Exit          (ExitCode (..), exitWith)

main :: IO ()
main = do
  model <- loadModel "data/lid.176.ftz"
  putStrLn $ "Loaded model with dimension " <> show (getDimension model)
  putStrLn $ "Word vector of 'stadig':"
  v <- getWordVector model "stadig"
  wordVectorData v >>= print
  let input = "og overalt gaar det tidligere undertrykte maal stadig fremad, trods brølene og trods dannelsen — akkurat som her"
      expLabel = "__label__no"
  res <- predictProbs model 2 0 input
  case res of
    (Prediction _ l) : _ | l == expLabel -> print "success" >>exitWith ExitSuccess
    _ -> do
      putStrLn $ "Test failed, expected " <> show expLabel <> " as top label, result, got:\n " <> show res <> " ... "
      exitWith (ExitFailure 1)


testWordVector model word = do
  v <- getWordVector model word
  print (FFI.vector_at v 0)
