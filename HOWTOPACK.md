
# How to pack

bin�t�H���_��obj�t�H���_�͎��O�ɍ폜���܂��B

```
dotnet clean
dotnet build -c Release -r win-x64 .\DeZero.NET\DeZero.NET.csproj
dotnet pack .\DeZero.NET\DeZero.NET.csproj -c Release -p:IncludeSymbols=true -p:SymbolPackageFormat=snupkg /p:PackageVersion=0.1.0 -p:RuntimeIdentifier=win-x64 --no-build
```