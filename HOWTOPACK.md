
# How to pack

binフォルダとobjフォルダは事前に削除します。

```
dotnet clean
dotnet build -c Release -r win-x64 .\DeZero.NET\DeZero.NET.csproj
dotnet pack .\DeZero.NET\DeZero.NET.csproj -c Release -p:IncludeSymbols=true -p:SymbolPackageFormat=snupkg /p:PackageVersion=0.11.0 -p:RuntimeIdentifier=win-x64 --no-build
```