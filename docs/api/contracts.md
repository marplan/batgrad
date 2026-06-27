# Contracts

Contracts define the shared column and metadata objects used by dataset configs,
processing stages, and generated outputs.


::: batgrad.contracts.mapping.MappingSpec
    options:
      heading_level: 3
      members:
        - with_alias
        - with_parser
        - with_values
        - values
        - matching_name


::: batgrad.contracts.metadata.MetadataLayout
    options:
      heading_level: 3
      members:
        - columns
        - values
        - with_optional

::: batgrad.contracts.metadata.StageLayout
    options:
      heading_level: 3
      members:
        - with_manifest
        - with_footer
