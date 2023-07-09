from trainer import Trainer
import logging
import os 
import torch
import gc
from utils import EmpathyDataset

# new_model_dir = 'output_empathyclass/logs'
# os.makedirs(new_model_dir, exist_ok=True)  # Create the directory if it doesn't exist

# # remove any existing handlers from the root loger
# [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]

# # configure logging to write to a file
# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s %(levelname)s %(message)s',
#     handlers=[
#         # output log to a file
#         logging.FileHandler(os.path.join(new_model_dir, 'trainlogs.log'))
#         ]
#     )

if __name__ == "__main__":    
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU index (0, 1, 2, etc.) or device UUID
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('Using GPU: {}'.format(torch.cuda.get_device_name(device)))
        # print('Using GPU: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        logging.info('Using CPU')
        # print('Using CPU')
    
    # rubbish collection
    gc.collect()
    torch.cuda.empty_cache()
    
    score_data_path = "data/new_utterances_with_fluency_score.csv"

    # train the model
    trainer = Trainer(batch_size=20, epochs=20, train_path=train_path, val_path=val_path, test_path=test_path, location='empathy_model/RoBERTa_empathy_2ft.pt')
    trainer.fit()
    
    # evaluate the model
    trainer.evaluate()
    
    # rubbish collection
    gc.collect()
    torch.cuda.empty_cache()
    
    # Finetune on EmpatheticPersonas dataset
    #we define the paths for train, val, test
    train_path = "empathy_dataset/second_ft/my_train.txt"
    test_path = "empathy_dataset/second_ft/my_test.txt"
    val_path = "empathy_dataset/second_ft/my_val.txt"
    
        # train the model
    trainer_second_ft = Trainer(batch_size=20, epochs=20, second_ft=True, train_path=train_path, val_path=val_path, test_path=test_path, location='empathy_model/RoBERTa_empathy_2ft_2.pt')
    trainer_second_ft.model.load_state_dict(torch.load('saved_models/best_model/best_model_first_ft.pt'))

    trainer_second_ft.fit()
    
    # evaluate the model
    trainer_second_ft.evaluate()
    
    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]