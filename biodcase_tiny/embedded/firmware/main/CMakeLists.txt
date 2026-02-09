idf_component_register(
    SRCS main.cpp metrics.cpp model.cpp esp_micro_profiler.cpp feature_extraction.cpp feature_config.cpp
    PRIV_REQUIRES spi_flash driver esp_timer
    INCLUDE_DIRS "")

    # Reduce the level of paranoia to be able to compile sources
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits)
