# Pycon 9 - Deep learning from zero to hero

Here both the slides and demos of my talk "Deep learning from zero to hero" at [pycon9 2018](https://www.pycon.it/)

## System requirements
The only dependence of the demos is docker. All demos run in the docker image [rucka/deeplearning](https://hub.docker.com/r/rucka/deeplearning/)

## The demos

- 1: The simpson transfer learning
	- Train the model with 30 epochs (accuracy ~50%): `./exec_cmd.sh /code/retrain.fast.sh` 
	- Train the model with 4000 epochs (accuracy ~90%): `./exec_cmd.sh /code/retrain.sh` 
	- Evaluate the fast model: `./exec_cmd.sh /code/evaluate.fast.sh /data/simpson/test_set/0.jpg`
	- Evaluate the accurated model: `./exec_cmd.sh /code/evaluate.sh /data/simpson/test_set/0.jpg`
- 2: Stock price regression forecast: execute the script `./run_book.sh`, open the jupyter home link showed in the command logs and select the notebook `1. stock regression.ipynb`

- 3: Stock trend classification forecast: execute the script `./run_book.sh`, open the jupyter home link showed in the command logs and select the notebook `2. stock classification.ipynb`. This notebook contains:
	- Multi layer perceptron network
	- Convolutional network
	- CNN vs MLP
	- Multi value classification

## Available scripts
- [exec_cmd.sh](exec_cmd.sh) to run a command inside the container e.g. `./exec_cmd.sh "/code/evaluate.fast.sh /data/simpson/test_set/0.jpg"` 

- [run_book.sh](run_book.sh) to run a jupyter notebook so you can write and test your code from your browser at http://localhost:8888. If you don't know jupyter, please take a look at the [official documentation](https://jupyter-notebook.readthedocs.io/en/stable/)

- [run_board.sh](run_board.sh) to run the tensorflow board and connect to it from the browser at http://localhost:6006. You can look at the [Tensorflow documentation](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for further informations

- [run_shell.sh](run_shell.sh) to open a container shell session 

## Slides
The slide markup has been compiled with [Deckset app for OSX](http://www.decksetapp.com). If you need, you can give a try of the trial version available at the [website](http://www.decksetapp.com/try.html).

The pdf slides are available [here](/slides/slides.pdf)

The slides are also published on [slideshare](https://www.slideshare.net/rucka/deep-learning-from-zero-to-hero).

## Abstract (italian)
Avete sentito parlare di Deep Learning ma credete che la teoria alla base sia troppo complessa? Non avete una laurea in matematica e statistica e pensate che il machine learning non faccia per voi? Niente paura: avrete solo bisogno di una conoscenze di base di Python.

Conoscete la regola dell’80/20? Con il 20% delle conoscenze potete raggiungere l’80% dei risultati: in questo talk vi mostrerò in modo pratico tramite delle demo - alcuni trucchi per costruire dei buoni modelli predittivi, evitando di perdere (tanto) tempo nella scelta dei tools e delle librerie necessarie al vostro scopo.

L’obbiettivo è fornirvi le basi pratiche con cui scegliere un modello di rete neurale, farne training e ottimizzarlo nel modo più adatto alla tipologia del problema che dovete affrontare.
