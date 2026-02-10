.PHONY: install uninstall reinstall clean-cache

install:
	UV_NO_CACHE=1 uv tool install . --force

uninstall:
	uv tool uninstall council

clean-cache:
	uv cache clean council 2>/dev/null || true

reinstall: uninstall clean-cache install
