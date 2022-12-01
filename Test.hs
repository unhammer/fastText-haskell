{-# LANGUAGE OverloadedStrings            #-}

module Main where

import Data.FastText
import System.Exit          (ExitCode (..), exitWith)

main :: IO ()
main = do
  model <- loadModel "data/lid.176.ftz"
  putStrLn $ "Loaded model with dimension " <> show (getDimension model)
  let input = "og overalt gaar det tidligere undertrykte maal stadig fremad, trods brølene og trods dannelsen — akkurat som her"
      expLabel = "__label__no"
  res1 <- predictBest model 0.0 input
  try1 <- case res1 of
    (Prediction _ l) | l == expLabel -> putStrLn ("Test 1 succeeded, got: " <> show res1) >> pure True
    _ -> do
      putStrLn $ "Test 1 failed, expected " <> show expLabel <> " as top label, result, got:\n " <> show res1 <> " ... "
      pure False
  res2 <- predictProbs model 2 0.0 input
  try2 <- case res2 of
    (Prediction _ l) : _ | l == expLabel -> putStrLn ("Test 2 succeeded, got: " <> show res2) >> pure True
    _ -> do
      putStrLn $ "Test 2 failed, expected " <> show expLabel <> " as top label, result, got:\n " <> show res2 <> " ... "
      pure False
  if and [try1, try2]
  then exitWith ExitSuccess
  else exitWith (ExitFailure 1)
