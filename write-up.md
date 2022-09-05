Project overview: This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self-driving car systems?
Set up: This section should contain a brief description of the steps to follow to run the code for this repository.
Dataset
Dataset Analysis: This section should contain a quantitative and qualitative description of the dataset. It should include images, charts, and other visualizations.
Cross-validation: This section should detail the cross-validation strategy and justify your approach.
Training


# Udacity - Self driving engineer nanodegree nd0013

## Project 1 - Computer vision

### Project overview

This project is meant to explore the modern techniques for object detection and teach us how to fine tune existing models for self-driving applications. This object detection technique allows the SD (self-driving) system to detect important parts of its environment and to enrich the understanding of its surrounding. This can be used to avoid pedestrians and cars, and to feed that information into another system that can predict what they will most likely do next, and plan accordingly.

### Set up

#### Build the docker image and run it
```shell
# Build the image
docker build -t project-dev -f Dockerfile .

# Run the image
docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -p 3002:3002 -p 6006:6006 -ti project-dev bash
```

#### Training time
Assuming you have all the files in the right locations, its time to train the model. You'll need to modify the `pipeline_new.config` in the `experiments/experiments1` directory with these file locations.

It should look like this:
```
train_input_reader {
  label_map_path: "label_map.pbtxt"
  tf_record_input_reader {
    input_path: "./splits/train/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
    ....
```

I went with a 96/2/2 split for train/val/test since there were so few files. For better result, I recommend running with a lot more files and a 80/10/10 split.

Now to run the training:
```
python experiments/model_main_tf2.py --model_dir=experiments/experiments1/ --pipeline_config_path=experiments/experiments1/pipeline_new.config
```

#### Running inference
```shell
# Export your model
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/experiment1/pipeline_new.config --trained_checkpoint_dir experiments/experiment1/ --output_directory experiments/experiment1/exported/

# Run a test inference and save the animation
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/experiment1/exported/saved_model --tf_record_path splits/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/experiment1/pipeline_new.config --output_path animation.gif
```

You should now have a file called `animation.gif` with your results!


### Dataset analysis


#
