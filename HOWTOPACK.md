
# How to pack

binフォルダとobjフォルダは事前に削除します。

```
dotnet clean
dotnet build -c Release -r win-x64 .\src\DeZero.NET\DeZero.NET.csproj
dotnet pack .\src\DeZero.NET\DeZero.NET.csproj -c Release -p:IncludeSymbols=true -p:SymbolPackageFormat=snupkg /p:PackageVersion=0.1.0 -p:RuntimeIdentifier=win-x64 --no-build
```