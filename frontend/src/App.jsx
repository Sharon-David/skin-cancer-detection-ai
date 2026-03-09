import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./App.css";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import DisclaimerBanner from "./components/DisclaimerBanner";
import Home from "./pages/Home";
import Detect from "./pages/Detect";
import About from "./pages/About";
import Resources from "./pages/Resources";

export default function App() {
  return (
    <BrowserRouter>
      <DisclaimerBanner />
      <Navbar />
      <main style={{ flex: 1, width: "100%" }}>
        <Routes>
          <Route path="/"          element={<Home />} />
          <Route path="/detect"    element={<Detect />} />
          <Route path="/about"     element={<About />} />
          <Route path="/resources" element={<Resources />} />
        </Routes>
      </main>
      <Footer />
    </BrowserRouter>
  );
}
