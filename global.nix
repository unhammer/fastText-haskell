let
  pkgs = import ./nixpkgs.nix {};

in
  pkgs.mkShell {
    packages = [
      pkgs.cabal-install
    ];
  }
