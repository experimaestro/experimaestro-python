# Logo / web-app icons
#
# The master logo (full artwork: graph + hand + "Experimaestro" word-mark) lives
# in the documentation. The web-app icons are derived from it by
# scripts/make-app-icon.py (drop the word-mark, recolour, crop). Run `make icons`
# after editing the master.

ICON_MASTER  := docs/source/img/icon.svg
ICON_DARK_BG := app/public/icon.svg     # lightened, vivid colours for dark surfaces (navbar-on-dark)
ICON_LIGHT_BG := app/public/favicon.svg # original brand colours for light surfaces (browser tab)
ICON_ADAPTIVE := docs/source/img/icon-adaptive.svg

.PHONY: icons
icons: $(ICON_DARK_BG) $(ICON_LIGHT_BG) $(ICON_ADAPTIVE) ## Regenerate the web-app icons from the master logo

$(ICON_DARK_BG): $(ICON_MASTER) scripts/make-app-icon.py
	uv run python scripts/make-app-icon.py $(ICON_MASTER) $@

$(ICON_LIGHT_BG): $(ICON_MASTER) scripts/make-app-icon.py
	uv run python scripts/make-app-icon.py $(ICON_MASTER) $@ --no-recolor

$(ICON_ADAPTIVE): $(ICON_MASTER) scripts/make-app-icon.py
	uv run python scripts/make-app-icon.py --adaptive $(ICON_MASTER) $@

