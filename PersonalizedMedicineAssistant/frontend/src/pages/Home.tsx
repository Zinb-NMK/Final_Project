import SymptomForm from "../components/SymptomForm";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100">
      <h1 className="text-3xl font-bold text-blue-600">Personalized Medicine Assistant</h1>
      <SymptomForm />
    </div>
  );
}
