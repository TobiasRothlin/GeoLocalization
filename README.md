# GeoLocalization

## Description
GeoLocalization is a project aimed at providing accurate geographical location services. It leverages various data sources and algorithms to pinpoint locations with high precision.

## Features
- High precision location detection

## Project Structure
 - Data Collection
    - Custom Code to download and process premade datasets


- Dataset Pipeline
    - Code to process and clean the data
    - Code to generate features from the data

## Usefull Commands
- Start new TMUX session:
    ```bash
    tmux new -s <session_name>
    ```
- Kill TMUX session:
    ```bash
    tmux kill-session -t <session_name>
    ```
- Attach to existing TMUX session:
    ```bash
    tmux attach -t <session_name>
    ```

- Stop Apptainer container:
    ```bash
    apptainer instance stop <session_name>
    ```

- List all runnig Apptainer Instances:
    ```bash
    apptainer instance list
    ```


## References
- GeoCLIP: clip-inspired alignment between locations and images for effective worldwide geo-localization
    - [Paper](https://arxiv.org/abs/2106.01861)
    - Published:
        - [NIPS '23: Proceedings of the 37th International Conference on Neural Information Processing Systems](https://dl.acm.org/doi/proceedings/10.5555/3666122) [Source](https://dl.acm.org/doi/10.5555/3666122.3666501#:~:text=To%20overcome%20these%20limitations,%20we%20propose%20GeoCLIP,%20a)

        - [NeurIPS 2023](https://dblp.org/db/conf/nips/neurips2023.html#CepedaNS23) [Source](https://dblp.org/search?q=GeoCLIP%3A+Clip-Inspired+Alignment+between+Locations+and+Images+for+Effective+Worldwide+Geo-localization)

        - [CoRR abs/2309.16020](https://dblp.org/db/journals/corr/corr2309.html#abs-2309-16020) [Source](https://dblp.org/search?q=GeoCLIP%3A+Clip-Inspired+Alignment+between+Locations+and+Images+for+Effective+Worldwide+Geo-localization)

        - [ Advances in Neural Information Processing Systems 36 (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023) [Source](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html)

- Learning Generalized Zero-Shot Learners for Open-Domain Image Geolocalization
    - [Paper](https://arxiv.org/abs/2302.00275)
    - Published:
        - [CoRR abs/2302.00275](https://dblp.org/db/journals/corr/corr2302.html#abs-2302-00275) [Source](https://dblp.org/search?q=Learning%20Generalized%20Zero-Shot%20Learners%20for%20Open-Domain%20Image%20Geolocalization)

        