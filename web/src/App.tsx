import { Navigate, Route, Routes } from "react-router-dom";
import { OverviewScreen } from "./screens/Overview";
import { CallsScreen } from "./screens/Calls";
import { ReviewScreen } from "./screens/review/Review";
import { PersonasScreen, PersonaEditorScreen } from "./screens/Personas";
import { ScenariosScreen, ScenarioEditorScreen } from "./screens/Scenarios";
import { RunLauncherScreen } from "./screens/RunLauncher";
import { RunProgressScreen } from "./screens/RunProgress";
import { TalkScreen } from "./screens/talk/Talk";
import { UploadScreen } from "./screens/Upload";
import { SettingsScreen } from "./screens/Settings";
import { CompareScreen } from "./screens/Compare";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<OverviewScreen />} />
      <Route path="/calls" element={<CallsScreen />} />
      <Route path="/calls/:id" element={<ReviewScreen />} />
      <Route path="/compare" element={<CompareScreen />} />
      <Route path="/scenarios" element={<ScenariosScreen />} />
      <Route path="/scenarios/new" element={<ScenarioEditorScreen />} />
      <Route path="/scenarios/:id" element={<ScenarioEditorScreen />} />
      <Route path="/personas" element={<PersonasScreen />} />
      <Route path="/personas/new" element={<PersonaEditorScreen />} />
      <Route path="/personas/:id" element={<PersonaEditorScreen />} />
      <Route path="/runs/new" element={<RunLauncherScreen />} />
      <Route path="/runs/:id" element={<RunProgressScreen />} />
      <Route path="/talk" element={<TalkScreen />} />
      <Route path="/upload" element={<UploadScreen />} />
      <Route path="/settings" element={<SettingsScreen />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
