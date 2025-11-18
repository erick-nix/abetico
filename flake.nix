{
  description = "Ambiente de desenvolvimento para o projeto Abetico";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = [
          pkgs.gobject-introspection
          pkgs.pkg-config
          pkgs.blueprint-compiler
        ];

        buildInputs = [
          pkgs.gtk4
          pkgs.libadwaita
          pkgs.gst_all_1.gstreamer
          (pkgs.python3.withPackages (
            p: with p; [
              pygobject3
              gst-python
              numpy
              matplotlib
              scikit-learn
              scipy
              pandas
              nuitka
            ]
          ))
        ];
      };
    };
}
