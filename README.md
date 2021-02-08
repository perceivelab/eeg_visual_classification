# EEG-Based Visual Classification Dataset

This dataset includes EEG data from 6 subjects. The recording protocol included 40 object classes with 50 images each, taken from the ImageNet dataset, giving a
total of 2,000 images. Visual stimuli were presented to the users in a block-based setting, with images of each class shown consecutively in a single sequence. Each image was shown for 0.5 seconds. A 10-second black screen (during which we kept recording EEG data) was presented between class blocks.
The collected dataset contains in total 11,964 segments (time intervals recording the response to each image); 36 have been excluded from the expected 6×2,000 = 12,000
segments due to low recording quality or subjects not looking at the screen, checked by using the eye movement data. Each EEG segment contains 128 channels, recorded for 0.5 seconds at 1 kHz sampling rate, represented as a 128×L matrix, with L about 500 being the number of samples contained in each segment on each
channel. The exact duration of each signal may vary, so we discarded the first 20 samples (20 ms) to reduce interference from the previous image and then cut the signal to a
common length of 440 samples (to account for signals with L < 500).
The dataset includes data already filtered in three frequency ranges: 14-70Hz, 5-95Hz and 55-95Hz. 


Download dataset
Link to dataset files: https://tinyurl.com/eeg-visual-classification

Citation
If you use this dataset, please cite the following works:

- S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909

- C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, Deep Learning Human Mind for Automated Visual Classification, International Conference on Computer Vision and Pattern Recognition, CVPR 2017 
