import torch
import pandas as pd
import numpy as np
import gc
import time
import argparse
import json
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from aum import AUMCalculator


def tokenize(text_df, tokenizer):
  token_ids = [] 
  attention_masks = []

  for i, text in text_df.iterrows():
    sentence1, sentence2 = text.values
    encoding_dict = tokenizer(sentence1, sentence2, padding="max_length", 
                              max_length=512, truncation=True, return_tensors="pt")
    token_ids.append(encoding_dict["input_ids"])
    attention_masks.append(encoding_dict["attention_mask"])

  return token_ids, attention_masks


def prepare_data(df, label_map, transformer, AUM_ind=False, AUM_save_dir=""):
  # Preprocessing and tensorizing
  labels = df.iloc[:,-1].map(label_map).tolist()
  labels = torch.tensor(labels)
  tot_num_examples = labels.shape[0]

  tokenizer = BertTokenizer.from_pretrained(transformer)
  text_df = df.iloc[:, 0:-1]
  token_ids, attention_masks = tokenize(text_df, tokenizer)
  token_ids = torch.cat(token_ids)
  attention_masks = torch.cat(attention_masks)

  # Mix in the indicator examples with a fake class
  if AUM_ind:
    fake_class = max(label_map.values())+1 
    tot_num_classes = len(label_map)+1
    num_real_examples = df.shape[0]
    num_indicators = num_real_examples//tot_num_classes
    
    indicator_labels = torch.full((num_indicators,), fake_class)
    labels = torch.cat([labels, indicator_labels])

    indicator_sample_idxes = np.random.choice(num_real_examples, num_indicators)
    
    indicator_token_ids = token_ids[indicator_sample_idxes]
    token_ids = torch.cat([token_ids, indicator_token_ids])

    indicator_attention_masks = attention_masks[indicator_sample_idxes]
    attention_masks = torch.cat([attention_masks, indicator_attention_masks])
  
    tot_num_examples += num_indicators
    indicators_idxes = np.arange(tot_num_examples-num_indicators, tot_num_examples)
    Path(AUM_save_dir).mkdir(parents=True, exist_ok=True)
    indicators_idxes.tofile(f"{AUM_save_dir}/indicator_idxes.txt", sep="\n")

  examples_idxes = torch.arange(tot_num_examples)

  dataset = TensorDataset(examples_idxes, token_ids, attention_masks,
                          labels)

  return dataset


def train_loop(dataloader, model, optimizer, device, aum_calc=None):
  """
  iterate over the training set and try to converge to optimal parameters
  """
  size = len(dataloader.dataset)
  for batch_idx, batch_data in enumerate(dataloader):
    gc.collect()   
    b_example_idxes = batch_data[0].tolist()
    b_token_ids, b_masks, b_labels = tuple(batch_data[i].to(device) for i in range(1,4))
    
    # Feed Forward
    output = model(b_token_ids, b_masks, labels=b_labels)

    # Backpropagation
    gc.collect()
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()

    # Calculating the AUM
    if aum_calc:
      aum_calc.update(logits=output.logits, targets=b_labels, 
                      sample_ids=b_example_idxes, 
                      grads=[0]*len(b_example_idxes)) # NOT SURE WHAT TO PUT FOR grads

    if batch_idx % 100 == 0:
      loss, current = output.loss.item(), batch_idx * len(b_labels)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
  # Setting the hyperparameters
  BATCH_SIZE = 32
  LEARN_R = 2e-5
  EPOCHS = 5

  # Take in the cmd line argument
  parser = argparse.ArgumentParser()
  parser.add_argument("transformer", type=str)
  parser.add_argument("save_model_fn", type=str)
  parser.add_argument("--AUM_ind", type=bool)
  parser.add_argument("--AUM_save_dir", type=str)
  cl_args = parser.parse_args()

  # Loading and preparing the data
  data_dir = "SciNLI_dataset"
  train_df = pd.read_csv(f"{data_dir}/train.csv").iloc[:, 2:]
  with open(f"{data_dir}/label_map.json") as js_file:
    label_map = json.load(js_file)

  train_dataset = prepare_data(train_df, label_map, cl_args.transformer, 
                               cl_args.AUM_ind, cl_args.AUM_save_dir)
  
  train_dataloader = DataLoader(train_dataset, shuffle = True, 
                                batch_size = BATCH_SIZE)

  # Building the model
  num_classes = len(label_map)
  if cl_args.AUM_ind:
    num_classes +=1

  model = BertForSequenceClassification.from_pretrained(
    cl_args.transformer,
    num_labels = num_classes,
    output_attentions = False,
    output_hidden_states = False
  )
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr = LEARN_R)
  if cl_args.AUM_ind:  
    aum_calc = AUMCalculator(save_dir=cl_args.AUM_save_dir, 
                             compressed=False)
  else:
    aum_calc = None

  # Training
  train_start = time.time()
  for e in range(EPOCHS):
    print(f"Epoch {e+1}\n-------------------------------")        
    model.train()   # Set model to training mode
    train_loop(train_dataloader, model, optimizer, device, aum_calc=aum_calc)
  aum_calc.finalize()
  print(f"\nElapsed time for training: {time.time()-train_start}")

  # Save the whole model
  torch.save(model, cl_args.save_model_fn)


if __name__ == "__main__":
  main()
  # [1] 1925043  