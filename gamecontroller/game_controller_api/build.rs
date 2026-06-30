use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut config = cbindgen::Config {
        after_includes: Some("namespace RoboCup { struct VAction; }".to_string()),
        sys_includes: vec!["cstdint".to_string()],
        no_includes: true,
        pragma_once: true,
        ..cbindgen::Config::default()
    };
    config.function.rename_args = cbindgen::RenameRule::CamelCase;
    config.structure.rename_fields = cbindgen::RenameRule::CamelCase;
    config.enumeration.rename_variants = cbindgen::RenameRule::CamelCase;

    cbindgen::Builder::new()
        .with_config(config)
        .with_crate(crate_dir)
        .with_namespace("RoboCup")
        .with_parse_deps(true)
        .with_parse_include(&["game_controller_core"])
        .generate()
        .expect("unable to generate bindings")
        .write_to_file("headers/GameController.h");
}
