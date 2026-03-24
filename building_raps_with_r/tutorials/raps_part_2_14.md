# Building RAPs with R - Part 2.14
Erika Duan
Invalid Date

- [What is Docker?](#what-is-docker)
  - [Docker images](#docker-images)
  - [Docker containers](#docker-containers)
  - [Dockerfiles](#dockerfiles)
  - [The Rocker project](#the-rocker-project)
- [Using Docker-based R and RStudio from the
  browser](#using-docker-based-r-and-rstudio-from-the-browser)
- [Other resources](#other-resources)

# What is Docker?

[Docker](https://www.docker.com/) is an open source tool that enables
maximal reproducibility by essentially allowing users to run the same
code from the same computer (or virtual environment). The RAPs textbook
offers a more comprehensive explanation of Docker
[here](https://raps-with-r.dev/repro_cont.html#what-is-docker).

Docker allows you to:

- Build a specific version of a Linux computer environment (also
  referred to as a specific operating system version).  
- Install a fixed version of R and all R packages that your project
  depends on.  
- Install a fixed version of other non-R applications that your project
  depends on.

Docker containers are lightweight by design as they:

- Share the same operating system kernel.  
- Start instantly using a layered filesystem (to minimise download
  time).

## Docker images

A Docker **image** is the binary archive of a specific virtual
environment.

R Docker images are built automatically by Docker Hub from public
(GitHub) source Dockerfiles. It can take a long time for each new Docker
image to be built, which is why central platforms for Docker image
hosting are so useful. Users can then download and run these binary
images that have been configured to ‘just work’.

## Docker containers

A Docker image only needs to be downloaded once and can then be run to
create a Docker **container** on demand. Multiple differently named
Docker containers can be run from the same Docker image. Because changes
to a Docker image’s virtual environment do not persist at run-time,
users need to create a **volume** (or shared folder between the Docker
container and host machine) to extract their Docker container outputs.

**Note:** `RUN` statements run at image build-time and `CMD` statements
run inside the container during run-time.

## Dockerfiles

A **Dockerfile** contains the code (or recipe) used to create the Docker
container. It is designed to be concise and easy to read by humans. For
project reproducibility, each Dockerfile should reference specific
software versions to always create the same Docker image.

Each new command in a Dockerfile contains an instruction to create a new
Docker image layer with a unique cryptographic hash. Each layer records
the differences from the layer below it, so Docker image layers resemble
a stack of filesystem changes (or deltas) that Docker combines together
to create a final image. The final image still resembles a single
unified filesystem.

Building from image layers is efficient as Docker can reuse cached base
layers when you locally rebuild an image from a modified Dockerfile.
Multiple images can also share common base layers available from a
central Docker images repository.

    FROM ubuntu:20.04           # Base layer: install base Linux Ubuntu operating system        
    RUN apt-get update          # New layer: update Linux package repository   
    RUN apt-get install python3 # New layer: install Python 3   
    COPY app.py /app/           # New layer: copy your Python script into the Docker container  

A [`docker-compose.yml`
file](https://stackoverflow.com/questions/29480099/whats-the-difference-between-docker-compose-vs-dockerfile)
also uses the instructions listed in your Dockerfile if the `build`
command exists in `docker-compose.yml`. It is additionally used to:

- Issue multiple Docker command line interface (CLI) commands more
  quickly  
- Start up multiple Docker containers and automatically connect them
  together  
- Easily map a port from your Docker container to your local machine

An example of a Docker and `docker-compose.yml` file can be found
[here](https://github.com/andrewheiss/silent-skywalk/tree/main/docker).

## The Rocker project

R Docker images with customised R environments are available through the
[Rocker project](https://rocker-project.org) and are listed
[here](https://rocker-project.org/images/versioned/rstudio.html). The
difference between these images include whether:

- The base image is based on a stable Debian Linux release
  i.e. `debian:jessie` or a Debian Linux pre-release
  i.e. `debian:testing`. Stable Debian releases are introduced once ~2
  years, so a stable Debian Linux environment can lag behind current R,
  RStudio or R package releases.  
- The RStudio Server is included. This allows users to code
  interactively using the RStudio IDE from a browser.  
- Additional report rendering tools and/or other R packages are
  included.

A stable and versioned image (which ships a specific R version) like
`rocker/tidyverse:3.4.1` is recommended for reproducing data analysis.
Versioned images offer the following features to ensure computator
environment reproducibility:

- A fixed version of R that is built from source.  
- The Debian Linux long term support release that was current when the R
  version was current.  
- The R package repository is set to the Posit Public Package Manager
  (PPPM) at a specific date. Users should use their project’s
  `renv.lock` file to install an identical package library inside their
  Docker image.

R Docker images can be extended through user modification of an existing
Dockerfile based on an appropriate R Docker image.

**Note:** From the [introduction to Rocker
paper](https://journal.r-project.org/articles/RJ-2017-065/index.html):
dependencies needed to compile R that are not required at runtime are
removed once R is installed, keeping the base images light-weight for
faster download times.

# Using Docker-based R and RStudio from the browser

We can run R and RStudio inside an existing Docker image built by the
Rocker Project following instructions from Andrew Heiss’ blog post on
[the
topic](https://www.andrewheiss.com/blog/2025/07/05/positron-ssh-docker/).

Andrew Hiess’s instructions on using [Docker
Compose](https://github.com/andrewheiss/lemon-lucifer?tab=readme-ov-file#method-1-docker-compose)

Example of Dockerfile and docker-compose.yml
(https://github.com/andrewheiss/silent-skywalk/tree/main/docker)

To test how using R inside a Docker image works, use instructions from
Andrew Heiss’ blog at
(https://www.andrewheiss.com/blog/2025/07/05/positron-ssh-docker/).

What it achieves:  
+ We can run R version 4.5.0 and access an RStudio IDE from
http://localhost:8787  
+ Our main project directory is ‘mounted’ or linked into the container.
So we can update all project code from inside the Docker image.

1.  Install repository https://github.com/andrewheiss/positron-docker#
2.  Install Docker Desktop on computer.
    https://docs.docker.com/desktop/setup/install/windows-install/
    Docker Desktop 4.55.0 was installed on my computer. This lets me
    download a Windows Subsystem for Linux.  
3.  If using Positron, ensure you install the Container Tools extension
    https://open-vsx.org/extension/ms-azuretools/vscode-containers The
    easiest way is to install it through the Extensions Pane through the
    Positron IDE.  
4.  Open a docker compose yml file and compose up (equivalent of typing
    docker compose -f docker-compose-basic.yml up -d into the
    terminal).  
5.  

Note from the docker compose yml file /project:/home/rstudio/project

- in your local repository, it takes the directory ./project and mounts
  that as home/rstudio/project inside your docker container’s RStudio
  IDE and container directory
- the magic is that any edits made in your docker container will then be
  saved in your local project directory on your local computer.  
- Because you are working inside a Docker image - many settings that you
  make and persist locally will disappear the next time you run your
  docker image (unless you make specific changes in your docker compose
  file).  
- Old way using docker compose: ‘I generally do all my coding and
  writing and analysis on my local computer and then try running it in
  the container at the end.’’

1.  A whole local Positron window can run on a remote server.

# Other resources

- Andrew Heiss’ blog post about [using Positron to run R inside a Docker
  image](https://www.andrewheiss.com/blog/2025/07/05/positron-ssh-docker/)
  through SSH  
- Chapter on [R production
  elements](https://datasciworkflows.netlify.app/chapters/elements_of_prod_code#operating-system-dependencies)
  from \[Data Science Workflows in R\] by Dean Marchiori  
- Talk on [establishing reproducibility standards in statistical
  consulting](https://deanmarchiori.github.io/biometrics-2025-talk/#/title-slide)
  by Dean Marchiori
