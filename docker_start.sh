# The password the Sidewalk websites send lives in ~/.sidewalk-ai-api.env (one line: export SIDEWALK_AI_API_KEY=...),
# readable only by you, and is loaded here so nobody has to remember it at deploy time. Refuse to start without it.
# The port is bound to 127.0.0.1 so the only way in is through the https proxy on this machine.
[ -f ~/.sidewalk-ai-api.env ] && source ~/.sidewalk-ai-api.env
: "${SIDEWALK_AI_API_KEY:?not set. Put 'export SIDEWALK_AI_API_KEY=<key>' in ~/.sidewalk-ai-api.env and chmod 600 it}"

# The lab's folder of already-downloaded panoramas, mounted read-only so we don't re-download from Google. The two
# paths must match: the mount puts the folder in the container and SIDEWALK_SCRAPES_DIR tells the app where it is.
PANOS_DIR=/m-makeabilitylab/makeabilitylab/sidewalk_panos/Panoramas

# Pass "dev" to run a test instance alongside prod: the dev image (from `./docker_build.sh dev`) on port 60655, named so
# it's easy to remove (`docker rm -f sidewalk-ai-api-dev`), and not restarted on reboot. It needs ~9-10 GB of VRAM on
# top of prod's. Reach it from another machine with `ssh -L 60655:127.0.0.1:60655 <server>`.
# Starting dev again replaces the running dev container.
case "$1" in
  "")
    IMAGE=sidewalk-ai-api; PORT=60654; RUN_OPTS="--restart unless-stopped" ;;
  dev)
    IMAGE=sidewalk-ai-api-dev; PORT=60655; RUN_OPTS="--name sidewalk-ai-api-dev"
    docker rm -f sidewalk-ai-api-dev 2>/dev/null ;;
  *)
    echo "Usage: $0 [dev]" >&2; exit 1 ;;
esac

docker run --gpus all --runtime nvidia -d -p 127.0.0.1:$PORT:5000 $RUN_OPTS \
      -e SIDEWALK_AI_API_KEY \
      -e SIDEWALK_SCRAPES_DIR=$PANOS_DIR \
      -v $PANOS_DIR:$PANOS_DIR:ro \
      $IMAGE
