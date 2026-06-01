; Inno Setup script for Flashback One35
; Built from PyInstaller onedir output.
;
; Two design choices worth flagging:
;
;   1. Versioned install / AppId. Each release installs into its own
;      versioned directory and registers a distinct AppId, so two
;      releases (e.g. 1.5.0-beta and 1.5.1) can coexist on the same
;      machine. The trade-off is that installing 1.5.1 does NOT replace
;      1.5.0 — users wanting a clean upgrade uninstall the old one
;      first. This is the explicit ask: parallel installs > automatic
;      replacement.
;
;   2. The optional uninstall-time "remove saved settings" prompt
;      targets ONE file: the schema file the current app version uses
;      (vibe_state_X_Y_Z.json under the per-user AppData dir). Pre-1.5
;      vibe_state.json is never touched, and future schema files (used
;      by later releases that happen to be installed in parallel) are
;      not touched either. The Qt app-data dir itself is left in place
;      because other releases of Flashback share it.

#define MyAppName "Flashback One35 v2"
#define MyAppExeName "Flashback One35.exe"
#define MyAppPublisher "Flashback"
#define MyAppURL "https://github.com/dothmos/flashback-editor"

; Version is injected by CI via /D flag: iscc /DMyAppVersion=0.1.0-beta7
#ifndef MyAppVersion
  #define MyAppVersion "dev"
#endif

; Schema-versioned settings filename. Must match core.vibe_state._FILE_NAME.
; Bumped on schema breaks, not on every app release.
#ifndef MySettingsFile
  #define MySettingsFile "vibe_state_1_5_0.json"
#endif

; Per-user app-data directory used by Qt's QStandardPaths.AppDataLocation,
; derived from organizationName / applicationName in main.py. Kept here as
; a single source of truth for the uninstall cleanup.
#define MyAppDataDir "{userappdata}\Flashback\Flashback One35 v2"

[Setup]
; AppId varies per version so each release is a distinct entry in
; Add/Remove Programs and gets its own uninstaller. Keep the base GUID
; stable so we stay in the same "product family" for any future tooling
; that wants to enumerate Flashback installs.
AppId={{B7E3F1A2-8C4D-4E5F-9A6B-1D2E3F4A5B6C}_v{#MyAppVersion}
AppName={#MyAppName} {#MyAppVersion}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName} {#MyAppVersion}
DefaultGroupName={#MyAppName} {#MyAppVersion}
OutputBaseFilename=Flashback-Windows-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\icons\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Bundle the entire PyInstaller onedir output
Source: "..\dist\Flashback One35\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName} {#MyAppVersion}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName} {#MyAppVersion}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName} {#MyAppVersion}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} {#MyAppVersion}"; Flags: nowait postinstall skipifsilent

[Code]
var
  RemoveSettings: Boolean;

function InitializeUninstall(): Boolean;
begin
  // One yes/no prompt instead of a checkbox so the uninstaller stays a
  // single dialog flow. Defaults to "No" so a misclick does not lose
  // data, and so accidentally rerunning the uninstaller on a still-used
  // schema (e.g. a parallel 1.5.x install) leaves settings alone.
  RemoveSettings := MsgBox(
    'Also remove saved Flashback settings for this version?' + #13#10 +
    #13#10 +
    'Only the settings file used by this release ({#MySettingsFile})' + #13#10 +
    'will be removed. Settings from older or other Flashback versions' + #13#10 +
    'installed on this machine will not be affected.' + #13#10 +
    #13#10 +
    'Default is No.',
    mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  SettingsPath: String;
begin
  if (CurUninstallStep = usPostUninstall) and RemoveSettings then
  begin
    SettingsPath := ExpandConstant('{#MyAppDataDir}\{#MySettingsFile}');
    if FileExists(SettingsPath) then
    begin
      if not DeleteFile(SettingsPath) then
        MsgBox('Could not remove ' + SettingsPath + '. You can delete it manually.',
               mbInformation, MB_OK);
    end;
    // Intentionally do NOT remove MyAppDataDir itself, the pre-1.5
    // vibe_state.json, or other schema-versioned files — those may
    // belong to a parallel Flashback install on the same machine.
  end;
end;
