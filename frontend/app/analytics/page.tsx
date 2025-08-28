"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

interface MLStatus {
  models_trained: number;
  total_models: number;
  status_by_model: Record<string, boolean>;
}

interface SourceStats {
  id: number;
  source_type: string;
  hostname: string;
  status: string;
  events_processed: number;
  events_failed: number;
  last_event_ts: string | null;
}

interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  last_trained: string;
}

export default function AnalyticsPage() {
  const [mlStatus, setMlStatus] = useState<MLStatus | null>(null);
  const [sources, setSources] = useState<SourceStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics[]>([
    {
      name: "Isolation Forest",
      accuracy: 0.87,
      precision: 0.84,
      recall: 0.89,
      f1_score: 0.86,
      last_trained: "2024-01-15T10:30:00Z"
    },
    {
      name: "LSTM Autoencoder",
      accuracy: 0.92,
      precision: 0.90,
      recall: 0.94,
      f1_score: 0.92,
      last_trained: "2024-01-15T10:30:00Z"
    }
  ]);

  // Simulated performance data
  const performanceData = [
    { name: 'Mon', detections: 12, false_positives: 2, accuracy: 85 },
    { name: 'Tue', detections: 19, false_positives: 1, accuracy: 89 },
    { name: 'Wed', detections: 8, false_positives: 0, accuracy: 92 },
    { name: 'Thu', detections: 15, false_positives: 3, accuracy: 86 },
    { name: 'Fri', detections: 22, false_positives: 2, accuracy: 90 },
    { name: 'Sat', detections: 6, false_positives: 1, accuracy: 88 },
    { name: 'Sun', detections: 9, false_positives: 0, accuracy: 94 }
  ];

  const threatDistribution = [
    { name: 'SSH Brute Force', value: 45, color: '#8884d8' },
    { name: 'Password Spray', value: 25, color: '#82ca9d' },
    { name: 'Port Scan', value: 20, color: '#ffc658' },
    { name: 'Other', value: 10, color: '#ff7300' }
  ];

  useEffect(() => {
    fetchMLStatus();
    fetchSources();
  }, []);

  const fetchMLStatus = async () => {
    try {
      const response = await fetch('/api/ml/status');
      const data = await response.json();
      if (data.success) {
        setMlStatus(data.metrics);
      }
    } catch (error) {
      console.error('Failed to fetch ML status:', error);
    }
  };

  const fetchSources = async () => {
    try {
      const response = await fetch('/api/sources');
      const data = await response.json();
      if (data.success) {
        setSources(data.sources);
      }
    } catch (error) {
      console.error('Failed to fetch sources:', error);
    } finally {
      setLoading(false);
    }
  };

  const retrainModels = async (modelType: string = 'ensemble') => {
    setRetraining(true);
    try {
      const response = await fetch('/api/ml/retrain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_type: modelType }),
      });

      const data = await response.json();
      if (data.success) {
        alert(`Successfully retrained ${modelType} models with ${data.training_data_size} samples`);
        await fetchMLStatus();
      } else {
        alert(`Retraining failed: ${data.message}`);
      }
    } catch (error) {
      console.error('Retraining failed:', error);
      alert('Retraining failed: Network error');
    } finally {
      setRetraining(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active': return 'bg-green-500';
      case 'inactive': return 'bg-red-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const formatTimestamp = (timestamp: string | null) => {
    if (!timestamp) return 'Never';
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">ML Analytics & Control</h1>
          <p className="text-gray-600">Machine learning model performance and tuning</p>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="sources">Data Sources</TabsTrigger>
          <TabsTrigger value="tuning">Tuning</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* System Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Models Trained</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {mlStatus?.models_trained || 0}/{mlStatus?.total_models || 0}
                </div>
                <p className="text-xs text-gray-600">Active ML models</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Data Sources</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {sources.filter(s => s.status === 'active').length}
                </div>
                <p className="text-xs text-gray-600">Active ingestion sources</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Events Today</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {sources.reduce((sum, s) => sum + s.events_processed, 0).toLocaleString()}
                </div>
                <p className="text-xs text-gray-600">Processed events</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Detection Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">89.3%</div>
                <p className="text-xs text-gray-600">ML accuracy</p>
              </CardContent>
            </Card>
          </div>

          {/* Performance Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Detection Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="detections" stroke="#8884d8" strokeWidth={2} />
                    <Line type="monotone" dataKey="false_positives" stroke="#ff7300" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Threat Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={threatDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {threatDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="models" className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Machine Learning Models</h2>
            <Button 
              onClick={() => retrainModels('ensemble')}
              disabled={retraining}
              className="bg-purple-600 hover:bg-purple-700"
            >
              {retraining ? 'Retraining...' : 'Retrain All Models'}
            </Button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {modelMetrics.map((model, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{model.name}</span>
                    <Badge variant={mlStatus?.status_by_model[model.name.toLowerCase().replace(' ', '_')] ? "default" : "secondary"}>
                      {mlStatus?.status_by_model[model.name.toLowerCase().replace(' ', '_')] ? 'Trained' : 'Untrained'}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Accuracy</span>
                      <span className="text-sm font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={model.accuracy * 100} className="h-2" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Precision</span>
                      <span className="text-sm font-medium">{(model.precision * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={model.precision * 100} className="h-2" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Recall</span>
                      <span className="text-sm font-medium">{(model.recall * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={model.recall * 100} className="h-2" />
                  </div>

                  <div className="text-xs text-gray-500">
                    Last trained: {formatTimestamp(model.last_trained)}
                  </div>

                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full"
                    onClick={() => retrainModels(model.name.toLowerCase().replace(' ', '_'))}
                    disabled={retraining}
                  >
                    Retrain Model
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="sources" className="space-y-6">
          <h2 className="text-xl font-semibold">Data Sources</h2>
          
          <div className="grid gap-4">
            {sources.map((source) => (
              <Card key={source.id}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(source.status)}`} />
                      <div>
                        <h3 className="font-medium">{source.source_type}</h3>
                        <p className="text-sm text-gray-600">{source.hostname}</p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-sm font-medium">
                        {source.events_processed.toLocaleString()} events
                      </div>
                      <div className="text-xs text-gray-600">
                        {source.events_failed} failed
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-sm">Last Event</div>
                      <div className="text-xs text-gray-600">
                        {formatTimestamp(source.last_event_ts)}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {sources.length === 0 && (
            <Alert>
              <AlertDescription>
                No data sources configured. Configure ingestion agents to start collecting data.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>

        <TabsContent value="tuning" className="space-y-6">
          <h2 className="text-xl font-semibold">Model Tuning</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Isolation Forest Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Contamination Threshold</Label>
                  <Slider
                    defaultValue={[0.1]}
                    max={0.3}
                    min={0.01}
                    step={0.01}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>0.01</span>
                    <span>Current: 0.10</span>
                    <span>0.30</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Number of Estimators</Label>
                  <Slider
                    defaultValue={[100]}
                    max={500}
                    min={50}
                    step={10}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>50</span>
                    <span>Current: 100</span>
                    <span>500</span>
                  </div>
                </div>

                <Button className="w-full">Apply Changes</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>LSTM Autoencoder Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Hidden Size</Label>
                  <Slider
                    defaultValue={[64]}
                    max={256}
                    min={32}
                    step={16}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>32</span>
                    <span>Current: 64</span>
                    <span>256</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Sequence Length</Label>
                  <Slider
                    defaultValue={[10]}
                    max={50}
                    min={5}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>5</span>
                    <span>Current: 10</span>
                    <span>50</span>
                  </div>
                </div>

                <Button className="w-full">Apply Changes</Button>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Detection Thresholds</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <Label>Low Risk</Label>
                    <Slider
                      defaultValue={[0.2]}
                      max={1}
                      min={0}
                      step={0.05}
                      className="w-full"
                    />
                    <div className="text-center text-xs text-gray-500">0.20</div>
                  </div>

                  <div className="space-y-2">
                    <Label>Medium Risk</Label>
                    <Slider
                      defaultValue={[0.5]}
                      max={1}
                      min={0}
                      step={0.05}
                      className="w-full"
                    />
                    <div className="text-center text-xs text-gray-500">0.50</div>
                  </div>

                  <div className="space-y-2">
                    <Label>High Risk</Label>
                    <Slider
                      defaultValue={[0.8]}
                      max={1}
                      min={0}
                      step={0.05}
                      className="w-full"
                    />
                    <div className="text-center text-xs text-gray-500">0.80</div>
                  </div>

                  <div className="space-y-2">
                    <Label>Critical Risk</Label>
                    <Slider
                      defaultValue={[0.95]}
                      max={1}
                      min={0}
                      step={0.05}
                      className="w-full"
                    />
                    <div className="text-center text-xs text-gray-500">0.95</div>
                  </div>
                </div>

                <Button className="w-full">Update Thresholds</Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
