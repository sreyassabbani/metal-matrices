# File structure
- In the future, `shaders/` will probably have two subdirectories. `compute/` and `graphics/`


# Structs `Matrix` and `Vector`
- These two are important pieces of this project (whatever it turns into).
- This API is built very _specifically_ to operate _generically_. It is really built for contiguous arrays `[T; M]` and `[[T; N]; M]`, making the API really efficient (I believe). And type safety is just beyond anything.
