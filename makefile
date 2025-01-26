# .metal files and corresponding .air files
METAL_FILES := $(wildcard shaders/*.metal)
AIR_FILES := $(METAL_FILES:shaders/%.metal=shaders/%.air)

# Final metallib binary
METALLIB_FILE := shaders/shaders.metallib

# Compile down to Metal IR
shaders/%.air: shaders/%.metal
	@echo "Compiling $< to $@"
	xcrun -sdk macosx metal -c $< -o $@

# Generate final binary
$(METALLIB_FILE): $(AIR_FILES)
	@echo "Creating $@ from $^"
	xcrun -sdk macosx metallib $^ -o $@

# Default target to build the .metallib file
all: $(METALLIB_FILE)

# Clean up generated files
.PHONY: clean
clean:
	@echo "Removing files: $(AIR_FILES) $(METALLIB_FILE)"
	@rm -f $(AIR_FILES) $(METALLIB_FILE)
