# Pass "dev" to build a separately tagged image for testing, so the prod image isn't overwritten.
case "$1" in
  "")  IMAGE=sidewalk-ai-api ;;
  dev) IMAGE=sidewalk-ai-api-dev ;;
  *)   echo "Usage: $0 [dev]" >&2; exit 1 ;;
esac

# Besides the plain name that docker_start.sh runs, tag each build with the API version, build date, and commit (e.g.
# sidewalk-ai-api:1.0.1-2026-10-14-fbb1947, plus "-dirty" if there were uncommitted changes), so an image the plain name
# has moved off of still says where it came from. List them with `docker images sidewalk-ai-api`.
REV=$(git rev-parse --short HEAD)
git diff --quiet HEAD || REV="$REV-dirty"
VERSION_TAG="$(cat API_VERSION)-$(date +%F)-$REV"

docker  build  -t  $IMAGE  -t  $IMAGE:$VERSION_TAG  --label  org.opencontainers.image.revision=$(git rev-parse HEAD)  .
