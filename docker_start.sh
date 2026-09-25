# Export SIDEWALK_AI_API_KEY (the password the Sidewalk websites send) before running this; it's passed through to the
# container. The port is bound to 127.0.0.1 so the only way in is through the https proxy on this machine.
# Refuse to start without a password (note that sudo drops exported variables).
: "${SIDEWALK_AI_API_KEY:?export SIDEWALK_AI_API_KEY first}"

# The lab's folder of already-downloaded panoramas, mounted read-only so we don't re-download from Google. The two
# paths must match: the mount puts the folder in the container and SIDEWALK_SCRAPES_DIR tells the app where it is.
PANOS_DIR=/m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas

docker run --gpus all --runtime nvidia -d -p 127.0.0.1:60654:5000 --restart unless-stopped \
      -e SIDEWALK_AI_API_KEY \
      -e SIDEWALK_SCRAPES_DIR=$PANOS_DIR \
      -v $PANOS_DIR:$PANOS_DIR:ro \
      sidewalk-ai-api
