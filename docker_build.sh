# Pass "dev" to build a separately tagged image for testing, so the prod image isn't overwritten.
case "$1" in
  "")  IMAGE=sidewalk-ai-api ;;
  dev) IMAGE=sidewalk-ai-api-dev ;;
  *)   echo "Usage: $0 [dev]" >&2; exit 1 ;;
esac

# Also tag each build with its version, build time, and commit (e.g. sidewalk-ai-api:1.0.1-2026-10-14-1803-fbb1947)
# so old builds can be told apart for rollback. "-dirty" means uncommitted files went into the image.
REV=$(git rev-parse --short HEAD)
[ -z "$(git status --porcelain)" ] || REV="$REV-dirty"
VERSION_TAG="$(cat API_VERSION)-$(date +%F-%H%M)-$REV"

docker  build  -t  $IMAGE  -t  $IMAGE:$VERSION_TAG  --label  org.opencontainers.image.revision=$(git rev-parse HEAD)  .
