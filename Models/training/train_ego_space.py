#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import random
from argparse import ArgumentParser
import sys
sys.path.append('..')
from data_utils.load_ego_space import LoadDataEgoSpace
from training.ego_space_trainer import EgoSpaceTrainer


def main():

    parser = ArgumentParser()
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path", help="root path where pytorch checkpoint file should be saved")
    parser.add_argument("-b", "--pretrained_checkpoint_path", dest="pretrained_checkpoint_path", help="path to SceneSeg weights file for pre-trained backbone")
    parser.add_argument('-t', "--test_images_save_path", dest="test_images_save_path", help="path to where visualizations from inference on test images are saved")
    parser.add_argument("-r", "--root", dest="root", help="root path to folder where training data is stored")
    parser.add_argument('-l', "--load_from_save", action='store_true', help="flag for whether model is being loaded from a Scene3D checkpoint file")
    args = parser.parse_args()

    # Root path
    root = args.root

    # Model save path
    model_save_root_path = args.model_save_root_path

    # Data paths
    # ZENSEACT
    zenseact_labels_filepath = root + 'zenseact/gt_masks/'
    zenseact_images_filepath = root + 'zenseact/images/'

    # MAPILLARY
    mapillary_labels_fileapath = root + 'Mapillary_Vistas/Mapillary_Vistas/gt_masks/'
    mapillary_images_fileapath = root + 'Mapillary_Vistas/Mapillary_Vistas/images/'

    # COMMA10K
    comma10k_labels_fileapath = root + 'comma10k/comma10k/gt_masks/'
    comma10k_images_fileapath = root + 'comma10k/comma10k/images/'

    # Test data
    test_images = root + 'test/'
    test_images_save_path = args.test_images_save_path

    # ZENSEACT - Data Loading
    zenseact_dataset = LoadDataEgoSpace(zenseact_labels_filepath, zenseact_images_filepath, 'ZENSEACT')
    zenseact_num_train_samples, zenseact_num_val_samples = zenseact_dataset.getItemCount()

    # MAPILLARY - Data Loading
    mapillary_dataset = LoadDataEgoSpace(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
    mapillary_num_train_samples, mapillary_num_val_samples = mapillary_dataset.getItemCount()

    # COMMA10K - Data Loading
    comma10k_dataset = LoadDataEgoSpace(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
    comma10k_num_train_samples, comma10k_num_val_samples = comma10k_dataset.getItemCount()

    # Total number of training samples
    total_train_samples = zenseact_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = zenseact_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Load from checkpoint
    load_from_checkpoint = False
    if(args.load_from_save):
        load_from_checkpoint = True

    # Pre-trained model checkpoint path
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    checkpoint_path = args.pretrained_checkpoint_path

    # Trainer Class
    trainer = 0
    if(load_from_checkpoint == False):
        trainer = EgoSpaceTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path)
    else:
        trainer = EgoSpaceTrainer(checkpoint_path=checkpoint_path, is_pretrained=True)

    trainer.zero_grad()

    # Total training epochs
    num_epochs = 6
    batch_size = 32

    # Epochs
    for epoch in range(0, num_epochs):

        # Printing epochs
        print('Epoch: ', epoch + 1)

        # Iterators for datasets
        zenseact_count = 0
        comma10k_count = 0
        mapillary_count = 0

        is_zenseact_complete = False
        is_mapillary_complete = False
        is_comma10k_complete = False

        data_list = []
        data_list.append('ZENSEACT')
        data_list.append('MAPILLARY')
        data_list.append('COMMA10K')
        random.shuffle(data_list)
        data_list_count = 0

        # Batch size schedule
        if(epoch == 1):
            batch_size = 16

        if(epoch == 2):
            batch_size = 8

        if(epoch == 3):
            batch_size = 5

        if(epoch >= 4):
            batch_size = 3

        # Loop through data
        for count in range(0, total_train_samples):

            # Log counter
            log_count = count + total_train_samples*epoch

            # Reset iterators
            if(zenseact_count == zenseact_num_train_samples and \
                is_zenseact_complete == False):
                is_zenseact_complete = True
                data_list.remove('ZENSEACT')

            if(mapillary_count == mapillary_num_train_samples and \
               is_mapillary_complete == False):
                is_mapillary_complete = True
                data_list.remove('MAPILLARY')

            if(comma10k_count == comma10k_num_train_samples and \
               is_comma10k_complete == False):
                is_comma10k_complete = True
                data_list.remove('COMMA10K')

            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if(data_list[data_list_count] == 'ZENSEACT' and \
               is_zenseact_complete == False):
                image, gt = zenseact_dataset.getItemTrain(zenseact_count)
                zenseact_count += 1

            if(data_list[data_list_count] == 'MAPILLARY' and \
               is_mapillary_complete == False):
                image, gt = mapillary_dataset.getItemTrain(mapillary_count)
                mapillary_count +=1

            if(data_list[data_list_count] == 'COMMA10K' and \
                is_comma10k_complete == False):
                image, gt = comma10k_dataset.getItemTrain(comma10k_count)
                comma10k_count += 1

            # Assign Data
            trainer.set_data(image, gt)

            # Augmenting Image
            trainer.apply_augmentations(is_train=False)

            # Converting to tensor and loading
            trainer.load_data()

            # Run model and calculate loss
            trainer.run_model()

            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if((count+1) % batch_size == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if((count+1) % 250 == 0):
                trainer.log_loss(log_count)

            # Logging Image to Tensor Board every 1000 steps
            if((count+1) % 1000 == 0):
                trainer.save_visualization(log_count)

            # Save model and run validation on entire validation
            # dataset after 10000 steps
            if((count+1) % 40000 == 0 or (count+1) == total_train_samples):

                # Save Model
                model_save_path = model_save_root_path + 'iter_' + \
                    str(count + total_train_samples*epoch) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'

                trainer.save_model(model_save_path)

                # Test and save visualization
                print('Testing')
                trainer.test(test_images, test_images_save_path, log_count)

                # Validate
                print('Validating')

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                # Overall IoU
                running_IoU = 0

                # No gradient calculation
                with torch.no_grad():

                    # ZENSEACT
                    for val_count in range(0, zenseact_num_val_samples):
                        image_val, gt_val = zenseact_dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score = trainer.validate(image_val, gt_val)

                        # Accumulate individual IoU scores for validation samples
                        running_IoU += IoU_score

                    # MAPILLARY
                    for val_count in range(0, mapillary_num_val_samples):
                        image_val, gt_val = mapillary_dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score = trainer.validate(image_val, gt_val)

                        # Accumulate individual IoU scores for validation samples
                        running_IoU += IoU_score

                    # COMMA10K
                    for val_count in range(0, comma10k_num_val_samples):
                        image_val, gt_val = comma10k_dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score = trainer.validate(image_val, gt_val)

                        # Accumulate individual IoU scores for validation samples
                        running_IoU += IoU_score

                    # Calculating average loss of complete validation set
                    mIoU = running_IoU/total_val_samples
                    print('mIoU: ', mIoU)

                    # Logging average validation loss to TensorBoard
                    trainer.log_IoU(mIoU, log_count)

                # Resetting model back to training
                trainer.set_train_mode()

            data_list_count += 1

    trainer.cleanup()


if __name__ == '__main__':
    main()
# %%
