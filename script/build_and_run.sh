#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="MLXLoRAStudio"
BUNDLE_ID="io.github.goekdeniz-guelmez.mlx-lora-studio"
MIN_SYSTEM_VERSION="14.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/MLX LoRA Studio.app"
APP_CONTENTS="$APP_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"
APP_BINARY="$APP_MACOS/$APP_NAME"
INFO_PLIST="$APP_CONTENTS/Info.plist"
DMG_VOLUME_NAME="MLX LoRA Studio"
DMG_BACKGROUND_SOURCE="$ROOT_DIR/Sources/Media/logo_ultra-wide.png"
DMG_BACKGROUND_NAME="logo_ultra-wide.png"
DMG_ICON_SOURCE="$ROOT_DIR/Sources/Media/logo.png"

pkill -x "$APP_NAME" >/dev/null 2>&1 || true

mkdir -p "$DIST_DIR"
BUILD_BINARY="$DIST_DIR/$APP_NAME"
swiftc \
  "$ROOT_DIR/Sources/MLXLoRAStudio/App/MLXLoRAStudioApp.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Models/TrainingModels.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Models/PythonEnvironment.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Support/MemoryEstimator.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Support/LiveMemoryMonitor.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Stores/AppStore.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/PythonJobRunner.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/PythonEnvironmentDiscovery.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/PythonEnvironmentProvisioner.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/ProviderModelCatalog.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/RunArchive.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/HFCacheScanner.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/IOGPUWiredLimitService.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Services/JobCompletionNotifier.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/ContentView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/OnboardingTourView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/SidebarView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/TrainingView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/SyntheticDataView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/HFUploadView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/AlgorithmGuideView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/RunsView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/SettingsView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/AboutView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/LiveRunPanel.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/LiveMetricsView.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/SharedControls.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/PythonPickerSection.swift" \
  "$ROOT_DIR/Sources/MLXLoRAStudio/Views/HFAssetPicker.swift" \
  -o "$BUILD_BINARY" \
  -framework SwiftUI \
  -framework AppKit \
  -framework UserNotifications

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_MACOS"
mkdir -p "$APP_CONTENTS/Resources"
cp "$BUILD_BINARY" "$APP_BINARY"
chmod +x "$APP_BINARY"

# Bundle the app icon. The source PNG lives at Sources/Media/logo.png;
# the build copies it into the app's Resources directory under the
# name "AppIcon" so the generated Info.plist's CFBundleIconFile can
# refer to it. AppKit picks it up at launch and scales it to the
# dock / Finder / Get Info sizes automatically.
cp "$ROOT_DIR/Sources/Media/logo.png" "$APP_CONTENTS/Resources/AppIcon.png"

# Bundle the Python runner and local mlx-lm-lora checkout so a copied
# or drag-installed .app can start training without the source repo
# still being present beside it.
SUPPORT_DIR="$APP_CONTENTS/Resources/StudioSupport"
mkdir -p "$SUPPORT_DIR"
cp -R "$ROOT_DIR/Backend" "$SUPPORT_DIR/Backend"
cp -R "$ROOT_DIR/vendor" "$SUPPORT_DIR/vendor"

# Single source of truth for the bundle version. Falls back to
# "0.0.0" if the file is missing so the plist still writes cleanly
# during a fresh clone.
APP_VERSION="$([[ -f "$ROOT_DIR/version" ]] && tr -d '[:space:]' < "$ROOT_DIR/version" || echo "0.0.0")"

cat >"$INFO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>
  <key>CFBundleName</key>
  <string>MLX LoRA Studio</string>
  <key>CFBundleDisplayName</key>
  <string>MLX LoRA Studio</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleIconName</key>
  <string>AppIcon</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>$APP_VERSION</string>
  <key>CFBundleVersion</key>
  <string>$APP_VERSION</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MIN_SYSTEM_VERSION</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
  <key>NSHumanReadableCopyright</key>
  <string>Created by Gökdeniz Gülmez</string>
</dict>
</plist>
PLIST

codesign_app() {
  /usr/bin/codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null
}

open_app() {
  /usr/bin/open -n "$APP_BUNDLE"
}

create_dmg_volume_icon() {
  local stage="$1"
  local iconset="$stage/.dmg-volume.iconset"
  rm -rf "$iconset"
  mkdir -p "$iconset"

  for size in 16 32 128 256 512; do
    /usr/bin/sips -z "$size" "$size" "$DMG_ICON_SOURCE" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    /usr/bin/sips -z "$((size * 2))" "$((size * 2))" "$DMG_ICON_SOURCE" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
  done

  /usr/bin/iconutil -c icns "$iconset" -o "$stage/.VolumeIcon.icns"
  rm -rf "$iconset"
}

configure_dmg_finder_view() {
  local image_path="$1"
  local mount_dir=""

  mount_dir="$(/usr/bin/hdiutil attach "$image_path" -readwrite -noverify -nobrowse \
    | /usr/bin/awk -F'\t' '/\/Volumes\// { print $NF; exit }')"
  if [[ -z "$mount_dir" ]]; then
    echo "Could not mount $image_path for Finder styling" >&2
    return 1
  fi

  if [[ -f "$mount_dir/.VolumeIcon.icns" ]]; then
    /usr/bin/SetFile -a V "$mount_dir/.VolumeIcon.icns"
    /usr/bin/SetFile -a C "$mount_dir"
  fi

  /usr/bin/osascript <<APPLESCRIPT
tell application "Finder"
  tell disk "$DMG_VOLUME_NAME"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {120, 120, 920, 520}
    set viewOptions to the icon view options of container window
    set arrangement of viewOptions to not arranged
    set icon size of viewOptions to 104
    set background picture of viewOptions to POSIX file "$mount_dir/.background/$DMG_BACKGROUND_NAME"
    set position of item "MLX LoRA Studio.app" of container window to {220, 250}
    set position of item "Applications" of container window to {580, 250}
    update without registering applications
    delay 1
    close
  end tell
end tell
APPLESCRIPT

  /usr/bin/hdiutil detach "$mount_dir" >/dev/null
}

# Build a redistributable `.dmg` of the just-compiled `.app`. The
# image contains the app bundle and a symlink to `/Applications` so
# the user can drag-install in one step. The version is read from
# `version` at the repo root (single source of truth for the bundle
# plist and the DMG filename); if missing, we fall back to "0.0.0".
build_dmg() {
  # $APP_VERSION was resolved above from the `version` file (or
  # defaulted to "0.0.0"), so the in-app version line and the DMG
  # filename always agree.
  local dmg_name="MLX-LoRA-Studio-${APP_VERSION}.dmg"
  local dmg_path="$DIST_DIR/$dmg_name"

  # Stage the bundle under $DIST_DIR itself (NOT mktemp -d). hdiutil
  # runs sandboxed and `mktemp -d` lands on /var/folders/.../T, where
  # hdiutil cannot create the intermediate sparse image and reports a
  # confusing "No space left on device" error. $DIST_DIR is a normal
  # user-writable directory so the workflow is reliable.
  local stage="$DIST_DIR/.dmg-stage"
  local rw_image="$DIST_DIR/.dmg-rw.dmg"
  rm -rf "$stage" "$rw_image"
  mkdir -p "$stage/.background"

  # Stage the bundle: the .app + a /Applications symlink so Finder
  # shows the standard drag-to-Applications layout when the DMG is
  # opened. We copy (not symlink) the .app so the DMG is self-
  # contained and works on a clean Mac.
  cp -R "$APP_BUNDLE" "$stage/"
  ln -s /Applications "$stage/Applications"
  cp "$DMG_BACKGROUND_SOURCE" "$stage/.background/$DMG_BACKGROUND_NAME"
  create_dmg_volume_icon "$stage"

  # The dev build already writes CFBundleShortVersionString and
  # CFBundleVersion into the bundle plist (from the heredoc above,
  # using the same $APP_VERSION), so no PlistBuddy stamp is needed
  # here. The DMG picks up the version from the .app that's being
  # staged, so the in-app label and the DMG filename can never
  # disagree.

  rm -f "$dmg_path"
  # 1. Make a read-write image from the stage dir.
  # 2. Convert it to a compressed read-only image.
  # `-fs HFS+` keeps the DMG mountable on every macOS the project
  # supports (macOS 14+); APFS read-only images are slightly
  # smaller but can't be opened on older releases.
  /usr/bin/hdiutil create -srcfolder "$stage" -volname "$DMG_VOLUME_NAME" -ov -fs HFS+ -format UDIF "$rw_image" >/dev/null
  configure_dmg_finder_view "$rw_image"
  /usr/bin/hdiutil convert "$rw_image" -format UDZO -imagekey zlib-level=9 -o "$dmg_path" >/dev/null

  rm -rf "$stage" "$rw_image"
  echo "Built $dmg_path"
}

case "$MODE" in
  run)
    codesign_app
    open_app
    ;;
  --debug|debug)
    codesign_app
    lldb -- "$APP_BINARY"
    ;;
  --logs|logs)
    codesign_app
    open_app
    /usr/bin/log stream --info --style compact --predicate "process == \"$APP_NAME\""
    ;;
  --telemetry|telemetry)
    codesign_app
    open_app
    /usr/bin/log stream --info --style compact --predicate "subsystem == \"$BUNDLE_ID\""
    ;;
  --verify|verify)
    codesign_app
    open_app
    sleep 1
    pgrep -x "$APP_NAME" >/dev/null
    ;;
  --package|package)
    codesign_app
    build_dmg
    ;;
  *)
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify|--package]" >&2
    exit 2
    ;;
esac
