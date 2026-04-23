# Build + runtime image for the NSE Options Engine.
# Needed on hosts where WDAC / Device Guard blocks unsigned native exes.
# Runs inside WSL2; GPU passthrough via `--gpus all`.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential dos2unix \
 && rm -rf /var/lib/apt/lists/*

COPY build.sh /usr/local/bin/build
RUN dos2unix /usr/local/bin/build && chmod +x /usr/local/bin/build

WORKDIR /app
