let
  nixpkgs-source = builtins.fetchTarball {
    url = https://github.com/NixOS/nixpkgs/archive/0a3e712e2de437ee8dc29dea643173fb10ad30c4.tar.gz;
  };
in
  import nixpkgs-source
