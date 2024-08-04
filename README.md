In the folder `src` one can find all the Python scripts.

DO NOT EDIT ANY FILES IN SRC EXCEPT FOR SAMPLER.PY AND ONLY DO SO WHERE COMMENTS SAY TO

## Generating dataset

```bash
python3 sampler.py --density <DENSITY>
```

Add the flag `--odd` to produce the odd set of tumbling rate. Run both with and
without the flag to generate the full dataset.

Read INITIALIZATION section of Summer_Project_Summary.pdf for instructions how you may want to chnage the data generated.

For access to example dataset with rolling of 250 output frames, go to https://drive.google.com/drive/folders/1Ui5bv50yO5iSqvI6gGw8PPbwQEzMs075?usp=share_link, but recommended to generate own as file sizes can be very large.

It is recommended to create a test set to keep for testing purposes only to not conflict with training sets fo models in the future.

## Model Training

Use modelTrainer.ipynb in notebooks to train a model on the data you have generated. You can edit this notebook to change the name of the model among other things you may want although i would recommend not to.

You can view the learning curves of these models by uncommenting the cell after training. Thnis will output the values to a text file that can be read by learningCurves.ipynb for plotting.

## Model Analysis

Use kernel comparison to compare the kernels of trained models to known filters adn see how the filters of different layers are outputted

You can then use modelBreakdown.ipynb with some test data to see how the kernels change an example input frame and the layer's output

## Data variations

Finally use modelComparison to load a test set and multiple pre-trained models to compare their accuracies for 3 different graphical outputs
