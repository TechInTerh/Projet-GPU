# Projet-GPU

## Build

Simply run:

```
make
```

## Run

videos url for testing: https://cloud.lrde.epita.fr/s/oE5t9TpweK3cDBj?

```
./main [--cpu][--gpu] filepath1 filepath2
```

You can use the Python scripts to display a video from different frames.
You can had a json generated previously to draw the rectangle around the
objects.

```
python run/run.py frames_folder output_folder json
```
