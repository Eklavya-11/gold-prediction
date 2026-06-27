"use client"

import * as React from "react"
import { ChevronDownIcon, TrendingUp, Activity, Globe, Info, Clock, CheckCircle2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Label } from "@/components/ui/label"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

const GoldPriceWidget = () => {
  return (
    <div className="bg-[#e6e9ef] p-8 rounded-2xl border border-[#ccd0da] shadow-sm w-full h-full flex flex-col transition-all duration-300">
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <Activity className="text-[#df8e1d] w-5 h-5" />
          <h2 className="text-[#4c4f69] text-lg font-medium tracking-wide">Market History</h2>
        </div>
        <span className="text-xs font-mono text-[#5c5f77] bg-[#ccd0da] px-2 py-1 rounded">LIVE</span>
      </div>
      <div className="flex-grow rounded-xl overflow-hidden bg-[#eff1f5] border border-[#ccd0da]">
        <iframe
          src="https://www.ashesh.com.np/gold/chart.php?type=0&unit=tola&range=5000&range=365"
          className="w-full h-full min-h-[350px] border-none mix-blend-multiply opacity-80"
          scrolling="no"
          title="Gold Price Chart"
        />
      </div>
    </div>
  )
}

export default function GoldPricePredictor() {
  const [open, setOpen] = React.useState(false)
  const [date, setDate] = React.useState<Date | undefined>(undefined)
  const [data, setData] = React.useState<any>(null)
  const [isLoading, setIsLoading] = React.useState(false)

  const fetchData = async () => {
    if (!date) return
    setIsLoading(true)
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          year: date.getFullYear(),
          month: date.getMonth() + 1,
          day: date.getDate(),
        }),
      })
      const result = await res.json()
      if (result.predicted_price_24k) {
        setData(result)
      } else {
        console.error("Prediction Error:", result.error)
        setData(null)
      }
    } catch (error) {
      console.error("Prediction fetch failed:", error)
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }

  const calculateError = (predicted: number, actual: number) => {
    const error = Math.abs(predicted - actual) / actual * 100
    return error.toFixed(2)
  }

  return (
    <div className="min-h-screen bg-[#eff1f5] text-[#4c4f69] py-16 px-4 sm:px-6 lg:px-8 font-sans selection:bg-[#df8e1d]/20">
      <div className="max-w-7xl mx-auto space-y-12">

        {/* Header section - Catppuccin Latte */}
        <div className="text-center space-y-4 mb-16">
          <h1 className="text-4xl md:text-5xl font-light tracking-tight text-[#4c4f69]">
            Forecast <span className="font-semibold text-[#df8e1d]">Gold</span>
          </h1>
          <p className="text-[#5c5f77] max-w-2xl mx-auto text-sm tracking-wide uppercase">
            Algorithmic Forecasting & Real-Time Validation
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* Left Column: Prediction Controls & Results */}
          <div className="lg:col-span-5 flex flex-col gap-8">

            {/* Input Form */}
            <div className="bg-[#e6e9ef] p-8 rounded-2xl border border-[#ccd0da] shadow-sm">
              <Label htmlFor="date" className="text-[#5c5f77] text-sm mb-3 block font-medium uppercase tracking-wider">
                Target Date
              </Label>
              <div className="flex flex-col gap-6">
                <Popover open={open} onOpenChange={setOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      id="date"
                      className="w-full justify-between h-14 bg-[#eff1f5] border-[#ccd0da] hover:bg-[#ccd0da] hover:text-[#4c4f69] text-[#4c4f69] font-light rounded-xl transition-all"
                    >
                      {date
                        ? date.toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" })
                        : <span className="text-[#6c6f85]">Select Date</span>
                      }
                      <ChevronDownIcon className="text-[#6c6f85] w-4 h-4" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0 bg-[#eff1f5] border-[#ccd0da] rounded-xl" align="start">
                    <Calendar
                      mode="single"
                      selected={date}
                      startMonth={new Date(2023, 0)}
                      endMonth={new Date(2100, 0)}
                      onSelect={(selectedDate) => {
                        setDate(selectedDate)
                        setOpen(false)
                      }}
                      className="bg-[#eff1f5] text-[#4c4f69] rounded-xl border-none"
                    />
                  </PopoverContent>
                </Popover>

                <Button
                  className="w-full h-14 bg-[#4c4f69] hover:bg-[#5c5f77] text-[#eff1f5] font-medium rounded-xl transition-all duration-300 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={fetchData}
                  disabled={isLoading || !date}
                >
                  {isLoading ? (
                    <span className="flex items-center gap-3 text-[#ccd0da]">
                      <div className="w-4 h-4 rounded-full border-2 border-[#ccd0da] border-t-transparent animate-spin" />
                      Computing...
                    </span>
                  ) : "Analyze Market"}
                </Button>
              </div>
            </div>

            {/* Results Section */}
            {data && (
              <div className="bg-[#e6e9ef] p-8 rounded-2xl border border-[#ccd0da] shadow-sm flex flex-col gap-8 animate-in slide-in-from-bottom-4 duration-700">

                {/* 24K and 22K Predictions */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-[#5c5f77] uppercase tracking-widest mb-1">24 Carat (999)</p>
                    <p className="text-2xl font-light text-[#df8e1d]">
                      ₹{data.predicted_price_24k.toLocaleString("en-IN")} <span className="text-xs text-[#6c6f85] ml-1">/10g</span>
                    </p>
                  </div>
                  <div className="border-l border-[#ccd0da] pl-4">
                    <p className="text-xs text-[#5c5f77] uppercase tracking-widest mb-1">22 Carat</p>
                    <p className="text-2xl font-light text-[#4c4f69]">
                      ₹{data.predicted_price_22k.toLocaleString("en-IN")} <span className="text-xs text-[#6c6f85] ml-1">/10g</span>
                    </p>
                  </div>
                </div>

                {/* Validation Panel (Only if historical) */}
                {data.actual_price_24k && (
                  <div className="bg-[#eff1f5] border border-[#40a02b]/30 p-5 rounded-xl flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-[#40a02b] uppercase tracking-widest flex items-center gap-2">
                        <CheckCircle2 className="w-3 h-3" /> Historic Validation
                      </span>
                      <span className="text-xs font-mono text-[#5c5f77]">Error: {calculateError(data.predicted_price_24k, data.actual_price_24k)}%</span>
                    </div>
                    <div className="flex justify-between items-end">
                      <div>
                        <p className="text-xs text-[#5c5f77] mb-1">Actual 24K</p>
                        <p className="text-lg text-[#4c4f69]">₹{data.actual_price_24k.toLocaleString("en-IN")}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-[#5c5f77] mb-1">Actual 22K</p>
                        <p className="text-lg text-[#5c5f77]">₹{data.actual_price_22k.toLocaleString("en-IN")}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Macro Factors */}
                <div className="pt-6 border-t border-[#ccd0da]">
                  <p className="text-xs text-[#5c5f77] uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Globe className="w-3 h-3" /> Underlying Factors
                  </p>
                  <div className="grid grid-cols-2 gap-y-4 gap-x-2">
                    <div>
                      <p className="text-xs text-[#6c6f85]">Global Spot</p>
                      <p className="text-sm font-mono text-[#4c4f69]">${data.features_used.Global_Gold_USD}</p>
                    </div>
                    <div>
                      <p className="text-xs text-[#6c6f85]">USD/INR</p>
                      <p className="text-sm font-mono text-[#4c4f69]">₹{data.features_used.USD_INR}</p>
                    </div>
                    <div>
                      <p className="text-xs text-[#6c6f85]">S&P 500</p>
                      <p className="text-sm font-mono text-[#4c4f69]">{data.features_used.SP500}</p>
                    </div>
                    <div>
                      <p className="text-xs text-[#6c6f85]">Crude Oil</p>
                      <p className="text-sm font-mono text-[#4c4f69]">${data.features_used.Crude_Oil_USD}</p>
                    </div>
                  </div>
                </div>

              </div>
            )}
          </div>

          {/* Right Column: Chart */}
          <div className="lg:col-span-7">
            <GoldPriceWidget />
          </div>

        </div>
      </div>
    </div>
  )
}
