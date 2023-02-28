import torch
import pandas as pd
import gc
import argparse
import json
from torch.utils.data import DataLoader
from train_SciNLI import prepare_data 


def test_loop(dataloader, model, device):
  """
  evaluate the model’s performance against the test data
  """
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for batch_idx, batch_data in enumerate(dataloader):
      gc.collect()   
      # b_example_idxes = batch_data[0].tolist()
      b_token_ids, b_masks, b_labels = tuple(batch_data[i].to(device) for i in range(1,4))

      pred = model(b_token_ids, b_masks, labels=b_labels)
      pred_class_ids = pred.logits.argmax(1)
      correct += (pred_class_ids == b_labels).type(torch.float).sum().item()
      test_loss += pred.loss.item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
  BATCH_SIZE = 32

  # Take in the cmd line argument
  parser = argparse.ArgumentParser()
  parser.add_argument("model_filename", type=str)
  parser.add_argument("transformer", type=str)
  cl_args = parser.parse_args()

  # Loading and preparing the data
  data_dir = "SciNLI_dataset"
  test_df = pd.read_csv(f"{data_dir}/test.csv").iloc[:, 1:]
  with open(f"{data_dir}/label_map.json") as js_file:
    label_map = json.load(js_file)

  test_dataset = prepare_data(test_df, label_map, cl_args.transformer)

  test_dataloader = DataLoader(test_dataset, shuffle = True, 
                               batch_size = BATCH_SIZE)

  # Loading the model
  model = torch.load(cl_args.model_filename)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Testing       
  model.eval()   # Set model to evaluation mode
  test_loop(test_dataloader, model, device)


if __name__ == "__main__":
  main()