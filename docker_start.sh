# The password the Sidewalk websites send lives in ~/.sidewalk-ai-api.env (one line: export SIDEWALK_AI_API_KEY=...),
# readable only by you, and is loaded here so nobody has to remember it at deploy time. Refuse to start without it.
# The port is bound to 127.0.0.1 so the only way in is through the https proxy on this machine.
[ -f ~/.sidewalk-ai-api.env ] && source ~/.sidewalk-ai-api.env
: "${SIDEWALK_AI_API_KEY:?not set. Put 'export SIDEWALK_AI_API_KEY=<key>' in ~/.sidewalk-ai-api.env and chmod 600 it}"

# The lab's folder of already-downloaded panoramas, mounted read-only so we don't re-download from Google. The two
# paths must match: the mount puts the folder in the container and SIDEWALK_SCRAPES_DIR tells the app where it is.
PANOS_DIR=/m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas

docker run --gpus all --runtime nvidia -d -p 127.0.0.1:60654:5000 --restart unless-stopped \
      -e SIDEWALK_AI_API_KEY \
      -e SIDEWALK_SCRAPES_DIR=$PANOS_DIR \
      -v $PANOS_DIR:$PANOS_DIR:ro \
      sidewalk-ai-api
